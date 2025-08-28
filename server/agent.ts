// Load environment variables
import 'dotenv/config'
// Import Google's Gemini chat model for AI conversation
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
// Import MongoDB client type for database operations
import { MongoClient } from "mongodb"
// Import MongoDB Atlas vector search for semantic search capabilities
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"

// Initialize Google Gemini chat model for customer service conversations
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 0.3, // Lower temperature for more consistent customer service responses
  apiKey: process.env.GOOGLE_API_KEY,
})

// Initialize Google embeddings for vector search
const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "embedding-001",
  apiKey: process.env.GOOGLE_API_KEY,
})

// Main function to handle customer queries and provide assistance
export async function callAgent(
  client: MongoClient,
  message: string,
  threadId: string
): Promise<string> {
  try {
    // Get database and collection references
    const db = client.db("inventory_database")
    const collection = db.collection("items")

    // Set up vector search for finding relevant products
    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
      collection,
      indexName: "vector_index",
      textKey: "summary",
      embeddingKey: "embedding",
    })

    // Perform semantic search to find relevant products based on user query
    const searchResults = await vectorStore.similaritySearch(message, 3)

    // Extract product information from search results
    const relevantProducts = searchResults.map(result => ({
      name: result.metadata.item_name,
      description: result.metadata.item_description,
      brand: result.metadata.brand,
      price: result.metadata.prices,
      categories: result.metadata.categories,
      summary: result.pageContent
    }))

    // Create context-aware prompt for the AI assistant
    const systemPrompt = `You are a helpful e-commerce customer service assistant. 
    You help customers find furniture products and answer their questions.
    
    Based on the customer's message: "${message}"
    
    Here are some relevant products from our inventory:
    ${relevantProducts.map((product, index) => 
      `${index + 1}. ${product.name} by ${product.brand}
         Price: $${product.price.sale_price} (Regular: $${product.price.full_price})
         Categories: ${product.categories.join(', ')}
         Description: ${product.description}`
    ).join('\n\n')}
    
    Please provide a helpful response to the customer. If they're looking for products, 
    recommend the most suitable ones from the list above. If they have other questions, 
    answer them in a friendly and professional manner.
    
    Keep your response concise but informative.`

    // Get AI response
    const response = await llm.invoke(systemPrompt)
    
    return response.content as string

  } catch (error) {
    console.error('Error in agent:', error)
    return "I'm sorry, I'm having trouble processing your request right now. Please try again later."
  }
}