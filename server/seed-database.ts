// Import Google's Gemini chat model and embeddings for AI text generation and vector creation
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
// Import structured output parser to ensure AI returns data in specific format
import { StructuredOutputParser } from "@langchain/core/output_parsers"
// Import MongoDB client for database connection
import { MongoClient } from "mongodb"
// Import MongoDB Atlas vector search for storing and searching embeddings
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
// Import Zod for data schema validation and type safety
import { z } from "zod"
// Load environment variables from .env file (API keys, connection strings)
import "dotenv/config"

// Create MongoDB client instance using connection string from environment variables
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)

// Initialize Google Gemini chat model for generating synthetic furniture data
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",  // Use Gemini 1.5 Flash model
  temperature: 0.7,               // Set creativity level (0.7 = moderately creative)
  apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment variables
})

// Define simplified schema for furniture item structure to avoid type recursion
const itemSchema = z.object({
  item_id: z.string(),
  item_name: z.string(),
  item_description: z.string(),
  brand: z.string(),
  manufacturer_country: z.string(),       // Simplified from nested address object
  full_price: z.number(),                 // Simplified from nested prices object
  sale_price: z.number(),
  categories: z.array(z.string()),
  notes: z.string(),
})

// Create TypeScript type from Zod schema for type safety
type Item = z.infer<typeof itemSchema>

// Create parser without complex nesting to avoid infinite type recursion
const parser = StructuredOutputParser.fromZodSchema(z.array(itemSchema))

// Function to create database and collection before seeding
async function setupDatabaseAndCollection(): Promise<void> {
  console.log("Setting up database and collection...")
  
  // Get reference to the inventory_database database
  const db = client.db("inventory_database")
  
  // Create the items collection if it doesn't exist
  const collections = await db.listCollections({ name: "items" }).toArray()
  
  if (collections.length === 0) {
    await db.createCollection("items")
    console.log("Created 'items' collection in 'inventory_database' database")
  } else {
    console.log("'items' collection already exists in 'inventory_database' database")
  }
}

// Function to create vector search index
async function createVectorSearchIndex(): Promise<void> {
  try {
    const db = client.db("inventory_database")
    const collection = db.collection("items")
    await collection.dropIndexes()
    const vectorSearchIdx = {
      name: "vector_index",
      type: "vectorSearch",
      definition: {
        "fields": [
          {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,
            "similarity": "cosine"
          }
        ]
      }
    }
    console.log("Creating vector search index...")
    await collection.createSearchIndex(vectorSearchIdx);

    console.log("Successfully created vector search index");
  } catch (e) {
    console.error('Failed to create vector search index:', e);
  }
}

async function generateSyntheticData(): Promise<Item[]> {
  // Create detailed prompt instructing AI to generate furniture store data (simplified fields)
  const prompt = `You are a helpful assistant that generates furniture store item data. Generate 3 furniture store items. Each record should include the following fields: item_id, item_name, item_description, brand, manufacturer_country, full_price, sale_price, categories, notes. Ensure variety in the data and realistic values.

  ${parser.getFormatInstructions()}`  // Add format instructions from parser

  // Log progress to console
  console.log("Generating synthetic data...")

  try {
    // Send prompt to AI and get response
    const response = await llm.invoke(prompt)
    // Parse AI response into structured array of Item objects
    const data = await parser.parse(response.content as string)
    console.log(`Successfully generated ${data.length} items`)
    return data
  } catch (error) {
    console.error("Error generating synthetic data:", error)
    throw error
  }
}

// Function to create a searchable text summary from furniture item data
async function createItemSummary(item: Item): Promise<string> {
  // Return Promise for async compatibility (though this function is synchronous)
  return new Promise((resolve) => {
    // Extract manufacturer country information
    const manufacturerDetails = `Made in ${item.manufacturer_country}`
    // Join all categories into comma-separated string
    const categories = item.categories.join(", ")
    // Create basic item information string
    const basicInfo = `${item.item_name} ${item.item_description} from the brand ${item.brand}`
    // Format pricing information
    const price = `At full price it costs: ${item.full_price} USD, On sale it costs: ${item.sale_price} USD`
    // Get additional notes
    const notes = item.notes

    // Combine all information into comprehensive summary for vector search
    const summary = `${basicInfo}. Manufacturer: ${manufacturerDetails}. Categories: ${categories}. Price: ${price}. Notes: ${notes}`

    // Resolve promise with complete summary
    resolve(summary)
  })
}

// Main function to populate database with AI-generated furniture data
async function seedDatabase(): Promise<void> {
  try {
    // Establish connection to MongoDB Atlas
    await client.connect()
    // Ping database to verify connection works
    await client.db("admin").command({ ping: 1 })
    // Log successful connection
    console.log("You successfully connected to MongoDB!")

    // Setup database and collection
    await setupDatabaseAndCollection()
    
    // Create vector search index
    await createVectorSearchIndex()

    // Get reference to specific database
    const db = client.db("inventory_database")
    // Get reference to items collection
    const collection = db.collection("items")

    // Clear existing data from collection (fresh start)
    await collection.deleteMany({})
    console.log("Cleared existing data from items collection")
    
    // Generate new synthetic furniture data using AI
    const syntheticData = await generateSyntheticData()

    // Process each item: create summary and prepare for vector storage
    const recordsWithSummaries = await Promise.all(
      syntheticData.map(async (record) => ({
        pageContent: await createItemSummary(record),  // Create searchable summary
        metadata: {...record},                         // Preserve original item data
      }))
    )
    
    // Create embeddings model once and reuse
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "text-embedding-004",
    })
    
    // Process records in smaller batches to avoid memory issues
    const batchSize = 1 // Process 1 item at a time to minimize memory usage
    for (let i = 0; i < recordsWithSummaries.length; i += batchSize) {
      const batch = recordsWithSummaries.slice(i, i + batchSize)
      console.log(`Processing item ${i + 1} of ${recordsWithSummaries.length}`)
      
      try {
        // Store batch with vector embeddings in MongoDB
        await MongoDBAtlasVectorSearch.fromDocuments(
          batch,                       // Process batch of records
          embeddings,                  // Reuse embeddings instance
          {
            collection,                // MongoDB collection reference
            indexName: "vector_index", // Name of vector search index
            textKey: "embedding_text", // Field name for searchable text
            embeddingKey: "embedding", // Field name for vector embeddings
          }
        )

        // Log progress for each batch
        batch.forEach(record => {
          console.log("Successfully processed & saved record:", record.metadata.item_id)
        })

        // Add delay between items to allow garbage collection
        await new Promise(resolve => setTimeout(resolve, 500))
        
        // Force garbage collection if available
        if (global.gc) {
          global.gc()
        }
        
      } catch (error) {
        console.error(`Error processing item ${i + 1}:`, error)
        throw error
      }
    }

    // Log completion of entire seeding process
    console.log("Database seeding completed")

  } catch (error) {
    // Log any errors that occur during database seeding
    console.error("Error seeding database:", error)
  } finally {
    // Always close database connection when finished (cleanup)
    await client.close()
  }
}

// Execute the database seeding function and handle any errors
seedDatabase().catch(console.error)