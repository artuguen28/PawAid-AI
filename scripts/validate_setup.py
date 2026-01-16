import sys
from pathlib import Path

def validate_chromadb():
    """Test ChromaDB basic functionality"""
    print("Testing ChromaDB...")
    try:
        import chromadb
        
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        
        # Add test document
        collection.add(
            documents=["Dogs should not eat chocolate"],
            metadatas=[{"category": "toxic_foods"}],
            ids=["doc1"]
        )
        
        # Query
        results = collection.query(
            query_texts=["chocolate poisoning"],
            n_results=1
        )
        
        print("✅ ChromaDB working correctly")
        print(f"   Retrieved: {results['documents'][0]}")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False


def validate_embeddings():
    """Test OpenAI embeddings (requires API key)"""
    print("\nTesting OpenAI Embeddings...")
    try:
        from langchain_openai import OpenAIEmbeddings  # FIXED IMPORT
        import os
        
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not found in environment")
            print("   Set it in .env file")
            return False
            
        embeddings = OpenAIEmbeddings()
        test_embedding = embeddings.embed_query("test query")
        
        print(f"✅ OpenAI Embeddings working")
        print(f"   Embedding dimension: {len(test_embedding)}")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False


def validate_langchain_chroma():
    """Test LangChain + ChromaDB integration"""
    print("\nTesting LangChain + ChromaDB integration...")
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_core.documents import Document  # FIXED IMPORT
        
        # Create test documents
        docs = [
            Document(page_content="Chocolate is toxic to dogs", 
                    metadata={"source": "vet_guide.pdf"}),
            Document(page_content="Cats cannot eat grapes",
                    metadata={"source": "toxic_foods.pdf"})
        ]
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="test_integration"
        )
        
        # Test retrieval
        results = vectorstore.similarity_search("dog chocolate", k=1)
        
        print("✅ LangChain + ChromaDB integration working")
        print(f"   Retrieved: {results[0].page_content}")
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False


def main():
    """Run all validation checks"""
    print("=" * 50)
    print("PawAid Copilot - Setup Validation")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    results = [
        validate_chromadb(),
        validate_embeddings(),
        validate_langchain_chroma()
    ]
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All validation checks passed!")
        print("Ready to start development.")
    else:
        print("❌ Some checks failed. Fix issues before proceeding.")
        sys.exit(1)
    print("=" * 50)


if __name__ == "__main__":
    main()