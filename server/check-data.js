const { MongoClient } = require('mongodb');
require('dotenv').config();

async function checkData() {
  const client = new MongoClient(process.env.MONGODB_ATLAS_URI);
  await client.connect();
  const db = client.db('inventory_database');
  const collection = db.collection('items');
  const items = await collection.find({}).toArray();
  console.log('Items in database:');
  items.forEach((item, i) => {
    console.log(`${i+1}. ${item.item_name} - Categories: ${JSON.stringify(item.categories)}`);
  });
  await client.close();
}

checkData().catch(console.error);
