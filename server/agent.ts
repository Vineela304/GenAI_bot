// Import required modules from LangChain ecosystem
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { AIMessage, HumanMessage } from "@langchain/core/messages"
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts"
import { StateGraph } from "@langchain/langgraph"
import { Annotation } from "@langchain/langgraph"
import { DynamicTool } from "@langchain/core/tools"
import { ToolNode } from "@langchain/langgraph/prebuilt"
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb"
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import { MongoClient } from "mongodb"
import "dotenv/config"

// Utility function to handle API rate limits with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn()
    } catch (error: any) {
      if (error.status === 429 && attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000)
        console.log(`Rate limit hit. Retrying in ${delay / 1000} seconds...`)
        await new Promise(resolve => setTimeout(resolve, delay))
        continue
      }
      throw error
    }
  }
  throw new Error("Max retries exceeded")
}

// Main function that creates and runs the AI agent
export async function callAgent(client: MongoClient, query: string, thread_id: string) {
  try {
    const dbName = "inventory_database"
    const db = client.db(dbName)
    const collection = db.collection("items")

    // Simplified state structure to avoid deep type instantiation
    const GraphState = Annotation.Root({
      messages: Annotation<any[]>({
        reducer: (x, y) => x.concat(y),
      }),
    })

    const itemLookupTool = new DynamicTool({
      name: "item_lookup",
      description: "Gathers furniture item details from the Inventory database. Input should be a JSON string with 'query' and optional 'n' fields.",
      func: async (input: string) => {
        try {
          // Parse input (could be JSON string or simple query)
          let query: string;
          let n = 10;
          
          try {
            const parsed = JSON.parse(input);
            query = parsed.query || input;
            n = parsed.n || 10;
          } catch {
            query = input;
          }

          console.log("Item lookup tool called with query:", query)
          const totalCount = await collection.countDocuments()
          console.log(`Total documents in collection: ${totalCount}`)

          if (totalCount === 0) {
            return JSON.stringify({
              error: "No items found in inventory",
              message: "The inventory database appears to be empty",
              count: 0
            })
          }

          const dbConfig = {
            collection,
            indexName: "vector_index",
            textKey: "embedding_text",
            embeddingKey: "embedding",
          }

          const vectorStore = new MongoDBAtlasVectorSearch(
            new GoogleGenerativeAIEmbeddings({
              apiKey: process.env.GOOGLE_API_KEY,
              model: "text-embedding-004",
            }),
            dbConfig
          )

          const result = await vectorStore.similaritySearchWithScore(query, n)
          if (result.length === 0) {
            const textResults = await collection.find({
              $or: [
                { item_name: { $regex: query, $options: 'i' } },
                { item_description: { $regex: query, $options: 'i' } },
                { categories: { $regex: query, $options: 'i' } },
                { embedding_text: { $regex: query, $options: 'i' } }
              ]
            }).limit(n).toArray()

            return JSON.stringify({
              results: textResults,
              searchType: "text",
              query,
              count: textResults.length
            })
          }

          return JSON.stringify({
            results: result,
            searchType: "vector",
            query,
            count: result.length
          })
        } catch (error: any) {
          return JSON.stringify({
            error: "Failed to search inventory",
            details: error.message,
            query: input
          })
        }
      }
    })

    const tools = [itemLookupTool]
    const toolNode = new ToolNode(tools) as any

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",
      temperature: 0,
      maxRetries: 0,
      apiKey: process.env.GOOGLE_API_KEY,
    }).bindTools(tools)

    function shouldContinue(state: any) {
      const messages = state.messages
      const lastMessage = messages[messages.length - 1] as AIMessage

      if (lastMessage.tool_calls?.length) {
        return "tools"
      }
      return "_end_"
    }

    async function callModel(state: any) {
      return retryWithBackoff(async () => {
        const prompt = ChatPromptTemplate.fromMessages([
          [
            "system",
            `You are a helpful E-commerce Chatbot Agent for a furniture store.

IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items.

Current time: {time}`,
          ],
          new MessagesPlaceholder("messages"),
        ])

        const formattedPrompt = await prompt.formatMessages({
          time: new Date().toISOString(),
          messages: state.messages,
        })

        const result = await model.invoke(formattedPrompt)
        return { messages: [result] }
      })
    }

    const workflow = new StateGraph(GraphState as any)
      .addNode("agent", callModel)
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", shouldContinue)
      .addEdge("tools", "agent")

    const checkpointer = new MongoDBSaver({ client, dbName })
    const app = workflow.compile({ checkpointer })

    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)],
      },
      {
        recursionLimit: 15,
        configurable: { thread_id }
      }
    )

    const response = finalState.messages[finalState.messages.length - 1].content
    console.log("Agent response:", response)

    return response
  } catch (error: any) {
    console.error("Error in callAgent:", error.message)

    if (error.status === 429) {
      throw new Error("Service temporarily unavailable due to rate limits. Please try again in a minute.")
    } else if (error.status === 401) {
      throw new Error("Authentication failed. Please check your API configuration.")
    } else {
      throw new Error(`Agent failed: ${error.message}`)
    }
  }
}