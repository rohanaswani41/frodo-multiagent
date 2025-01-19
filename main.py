import os
from agents import MetaAgent
from embeddings import ProductProcessor
from db import Order, session, load_products_data, Product, ProductReview
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPEN_AI_API_KEY = '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'


# Example usage
if __name__ == "__main__":

    product_processor = ProductProcessor()
    input_products = './data/product_data.json'
    output_prod_embeddings = './product_output_embeddings.json'

    input_reviews = './data/reviews_data.json'
    output_review_embeddings = './reviews_output_embeddings.json'

    # Process product data
    json_data = product_processor.load_json(input_products)
    chunks = product_processor.product_extract_and_chunk(json_data)
    embeddings = product_processor.generate_embeddings(chunks)
    product_processor.save_embeddings(embeddings, chunks, output_prod_embeddings)
    print(f"Successfully processed and saved embeddings to {output_prod_embeddings}")

    json_data = product_processor.load_json(input_reviews)
    load_products_data(json_data)
        
    meta_agent = MetaAgent(
        openai_api_key=OPEN_AI_API_KEY,
        product_embeddings=output_prod_embeddings,
        order_orm = Order,
        db_session = session
    )

    # queries = [
    #     "Show me reviews for Product Ultra Comfort Mattress",
    #     "Can you tell me order status for the order id 26301759-53a7-45b4-9c9e-15c817522694?",
    #     "Compare the memory foam mattress with the latex mattress"
    # ]

    # for query in queries:
    #     print(f"\nQuery: {query}")
    #     responses = meta_agent.process_query(query)
    #     print(f"Response: {responses}")

    print("Frodo: How can I help you today ?")
    while True:
        inp_question = input("User:")
        print(f"""Frodo: {meta_agent.process_query(inp_question)}""")