import sys
import os
import asyncio
from pathlib import Path

# Add project root and myenv site-packages to sys.path
project_root = Path(__file__).resolve().parent
site_packages = project_root / "myenv" / "lib" / "python3.14" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingest import ingest_documents
from src.ingest_rules import ingest_rules
from src.excel_evaluator import ExcelTestEvaluator
from src.rag_chain import RAGChatbot
from src.test_eval_agent import TestEvalAgent
import src.config as config


def print_banner():
    print("=" * 60)
    print(" 🤖  RAG CHATBOT & AUTOMATED TEST EVALUATION ENGINE")
    print("=" * 60)
    print(" Commands:")
    print("  - Type your question to query your indexed documents")
    print("  - '/eval-excel'  : Run full automatic evaluation on Excel test cases")
    print("  - '/ingest-rules': Index/Update chatbot evaluation rules & command lists")
    print("  - '/eval'        : Interactive single test case evaluation & RCA")
    print("  - '/ingest'      : Re-index knowledge documents in data/")
    print("  - '/clear'       : Reset conversation chat history")
    print("  - '/exit'        : Quit engine")
    print("=" * 60 + "\n")


def main():
    print_banner()

    # Check if vector DB exists, offer ingestion if missing
    if not os.path.exists(config.CHROMA_DB_DIR):
        print(f"⚠️ Vector database not found at '{config.CHROMA_DB_DIR}'.")
        print("Running initial document ingestion from 'data/'...\n")
        ingest_documents()
        print("-" * 60)

    if not os.path.exists(config.RULES_CHROMA_DB_DIR):
        print(f"⚠️ Rules vector database not found at '{config.RULES_CHROMA_DB_DIR}'.")
        print("Indexing chatbot evaluation rules...\n")
        ingest_rules()
        print("-" * 60)

    try:
        chatbot = RAGChatbot()
        eval_agent = TestEvalAgent()
    except Exception as e:
        print(f"❌ Error initializing RAG Chatbot: {e}")
        print("💡 Hint: Ensure your OPENAI_API_KEY is configured in .env and run '/ingest'.")
        sys.exit(1)

    print("✅ RAG Chatbot & Automated Test Evaluation Engine ready!\n")

    while True:
        try:
            user_input = input(" You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                print("\nGoodbye! 👋\n")
                break

            if user_input.lower() == "/ingest":
                print("\n🔄 Re-indexing documents...")
                ingest_documents()
                chatbot = RAGChatbot()
                print("✅ Ingestion complete! Chatbot updated.\n")
                continue

            if user_input.lower() == "/ingest-rules":
                print("\n🔄 Re-indexing evaluation rules and command lists...")
                ingest_rules()
                print("✅ Rules vector store updated!\n")
                continue

            if user_input.lower() == "/eval-excel":
                print("\n📊 --- AUTOMATED EXCEL TEST CASE EVALUATION ---")
                filepath = input(" Enter Excel file path (press Enter for 'data/kết quảOM8NP.xlsx'): ").strip()
                if not filepath:
                    filepath = "data/kết quảOM8NP.xlsx"
                
                try:
                    evaluator = ExcelTestEvaluator()
                    output_file = evaluator.evaluate_file(filepath)
                    print(f"✅ Full automated evaluation finished. Results saved to: {output_file}\n")
                except Exception as e:
                    print(f"❌ Failed to evaluate Excel file: {e}\n")
                continue

            if user_input.lower() == "/clear":
                chatbot.clear_history()
                eval_agent.clear_history()
                print("🧹 Conversation history cleared.\n")
                continue

            if user_input.lower() == "/eval":
                print("\n🧪 --- TEST RESULT EVALUATION MODE ---")
                test_name = input(" Enter Test Name/ID: ").strip() or "TC_Execution"
                expected = input(" Enter Expected Behavior: ").strip()
                actual = input(" Enter Actual Log / Output: ").strip()
                error_log = input(" Enter Error Log / Traceback (optional): ").strip()

                eval_query = (
                    f"Evaluate test '{test_name}'. "
                    f"Expected: '{expected}'. "
                    f"Actual: '{actual}'. "
                    f"Error log: '{error_log}'."
                )

                print("\n🔄 Analyzing test results and performing Root Cause Analysis...\n")
                res = asyncio.run(eval_agent.process_message(eval_query))
                print("\n🤖 Evaluation Result:")
                print(res["reply"])
                print("-" * 60 + "\n")
                continue

            # Query the RAG engine
            result = chatbot.answer_question(user_input)

            print("\n🤖 Bot:")
            print(result["answer"])
            
            if result["sources"]:
                print("\n📚 Sources referenced:")
                for source in result["sources"]:
                    print(f"  • {source}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}\n")



if __name__ == "__main__":
    main()
