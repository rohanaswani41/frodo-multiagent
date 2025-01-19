from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages.ai import AIMessage
from embeddings import EmbeddingSearcher
from openai import OpenAI
import openai
import json
import uuid
from datetime import datetime, timedelta
from db import Product, ProductReview

# State and data structures
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    context: Dict[str, Any]
    intent: str
    product_data: Optional[Dict[str, Any]]
    comparison_data: Optional[List[Dict[str, Any]]]

class ProductDetailsAgent:
    def __init__(
        self, 
        embedding_file: str,
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str = None
    ):
        """Initialize the Product Details Agent with necessary components."""
        self.embeddings_searcher = EmbeddingSearcher(embedding_file, model_name)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph processing graph."""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("IdentifyIntent", self._identify_intent)
        graph_builder.add_node("RetrieveContext", self._retrieve_context)
        graph_builder.add_node("GenerateResponse", self._generate_response)
        graph_builder.add_node("CompareProducts", self._compare_products)
        
        # Define routing function
        def route_by_intent(state: State) -> str:
            intent_mapping = {
                "product_info": "RetrieveContext",
                "product_comparison": "CompareProducts",
            }
            return intent_mapping.get(state['intent'], "RetrieveContext")

        # Add edges
        graph_builder.add_edge(START, "IdentifyIntent")
        graph_builder.add_conditional_edges(
            "IdentifyIntent",
            route_by_intent
        )
        graph_builder.add_edge("RetrieveContext", "GenerateResponse")
        graph_builder.add_edge("CompareProducts", "GenerateResponse")
        graph_builder.add_edge("GenerateResponse", END)

        return graph_builder.compile()

    def _identify_intent(self, state: State) -> State:
        """Identify the user's intent from their query."""
        messages = [
            {"role": "system", "content": """
            Identify the intent of the user's query about products. Possible intents are:
            - product_info: Questions about specific product details
            - product_comparison: Requests to compare multiple products
            Respond with just the intent identifier.
            """},
            {"role": "user", "content": state['messages'][-1].content}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        state['intent'] = response.choices[0].message.content.strip()
        return state

    def _retrieve_context(self, state: State) -> State:
        """Retrieve relevant product information using RAG."""

        query = state['messages'][-1].content
        search_results = self.embeddings_searcher.search(query, top_k=3)
        
        context = []
        for result in search_results:
            context.append(result.text)
            
        state['context']['retrieved_info'] = "\n".join(context)
        return state

    def _compare_products(self, state: State) -> State:
        """Handle product comparison requests."""
        # Extract product names from the query
        messages = [
            {"role": "system", "content": "Extract the product names to compare from the query. Return as a comma-separated list."},
            {"role": "user", "content": state['messages'][-1].content}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        product_names = response.choices[0].message.content.split(',')
        product_names = [name.strip() for name in product_names]
        
        # Retrieve details for each product
        comparison_data = []
        for product_name in product_names:
            product_info = self.embeddings_searcher.get_product_details(product_name)
            if product_info:
                comparison_data.append(product_info)
        
        state['comparison_data'] = comparison_data
        return state

    def _generate_response(self, state: State) -> State:
        """Generate natural language response based on retrieved information."""
        if state['intent'] == 'product_comparison' and state.get('comparison_data'):
            prompt = self._create_comparison_prompt(state['comparison_data'])
        else:
            prompt = self._create_info_prompt(state['context'].get('retrieved_info', ''))
            
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": state['messages'][-1].content}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        state['messages'].append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return state

    def _create_comparison_prompt(self, comparison_data: List[Dict[str, Any]]) -> str:
        """Create a prompt for product comparison."""
        return f"""
        Compare the following products based on their features and specifications:
        {json.dumps(comparison_data, indent=2)}
        
        Provide a detailed comparison highlighting:
        1. Key similarities and differences
        2. Unique features of each product
        3. Relative advantages and disadvantages
        4. Price comparison if available
        
        Format the response in a clear, easy-to-read manner.
        """

    def _create_info_prompt(self, context: str) -> str:
        """Create a prompt for product information."""
        return f"""
        Using the following product information:
        {context}
        
        Provide a detailed and natural response that:
        1. Directly addresses the user's question
        2. Highlights relevant features and specifications
        3. Includes any important caveats or limitations
        4. Maintains a helpful and informative tone
        
        Only include information that is present in the provided context.
        """

    def process_query(self, query: str) -> List[str]:
        """Process a user query and return responses."""
        initial_state: State = {
            'messages': [{"role": "user", "content": query}],
            'context': {},
            'intent': '',
            'product_data': None,
            'comparison_data': None
        }
        
        final_state = self.graph.invoke(initial_state)

        return [
            msg.content
            for msg in final_state['messages'] 
            if isinstance(msg, AIMessage)
        ][0]

class ReviewState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    context: Dict[str, Any]
    intent: str
    reviews: Optional[List[Dict[str, Any]]]
    sentiment_filter: Optional[str]
    product_name: Optional[str]

@dataclass
class Review:
    reviewer: str
    comment: str
    rating: int
    sentiment: str
    product_name: str
    metadata: Optional[Dict[str, Any]] = None

class ProductReviewsAgent:
    def __init__(
        self,
        db_session,
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None
    ):
        """Initialize the Product Reviews Agent."""
        self.db_session = db_session
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph processing graph."""
        graph_builder = StateGraph(ReviewState)
        
        # Add nodes
        graph_builder.add_node("ParseQuery", self._parse_query)
        graph_builder.add_node("FetchReviews", self._fetch_reviews)
        graph_builder.add_node("AnalyzeReviews", self._analyze_reviews)
        graph_builder.add_node("GenerateResponse", self._generate_response)
        
        # Add edges
        graph_builder.add_edge(START, "ParseQuery")
        graph_builder.add_edge("ParseQuery", "FetchReviews")
        graph_builder.add_edge("FetchReviews", "AnalyzeReviews")
        graph_builder.add_edge("AnalyzeReviews", "GenerateResponse")
        graph_builder.add_edge("GenerateResponse", END)
        
        return graph_builder.compile()

    def _parse_query(self, state: ReviewState) -> ReviewState:
        """Parse the user query to extract product name and sentiment filter."""
        messages = [
            {"role": "system", "content": """
            Analyze the query and extract:
            1. Product name
            2. Sentiment filter (positive, negative, or neutral)
            Return as JSON with keys 'product_name' and 'sentiment_filter'
            """},
            {"role": "user", "content": state['messages'][-1].content}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
            response_format={ "type": "json_object" }
        )
        
        parsed = json.loads(response.choices[0].message.content)
        state['product_name'] = parsed.get('product_name')
        state['sentiment_filter'] = parsed.get('sentiment_filter', 'all')
        return state

    def _fetch_reviews(self, state: ReviewState) -> ReviewState:
        """Fetch reviews for the specified product."""

        if state['product_name']:

            product = self.db_session.query(Product).filter(Product.name == state['product_name']).first()
            
            if not product:
                state['reviews'] = [{'text':f"No product found with name '{state['product_name']}'"}]
            
            # Fetch all reviews for the product
            reviews = self.db_session.query(ProductReview).filter(ProductReview.product_id==product.id).all()
            
            # Format and return the reviews
            state['reviews'] = [{"text": review.review_text} for review in reviews]
        return state

    def _analyze_reviews(self, state: ReviewState) -> ReviewState:
        """Analyze the filtered reviews to generate insights."""
        if not state['reviews']:
            return state
        
        context = []
        for result in state['reviews']:
            context.append(result['text'])

        messages = [
            {"role": "system", "content": """
            Analyze these product reviews and provide response to the query provided by the user.
            """},
            {"role": "user", "content": "Set of Reviews \n".join(context)},
            {"role": "user", "content": f"User Query: {state['messages'][0].content}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )

        state['context']['analysis'] = response.choices[0].message.content
        return state

    def _generate_response(self, state: ReviewState) -> ReviewState:
        """Generate a natural language response about the reviews."""
        if not state['reviews']:
            state['messages'].append({
                "role": "assistant",
                "content": f"I couldn't find any reviews for {state['product_name']}."
            })
            return state
            
        analysis = state['context'].get('analysis', {})
        review_count = len(state['reviews'])
        
        response_template = f"""
        I found {review_count} reviews for {state['product_name']}.
        
        {analysis}
        """
        
        state['messages'].append({
            "role": "assistant",
            "content": response_template.strip()
        })
        return state


    def process_query(self, query: str) -> List[str]:
        """Process a review-related query and return responses."""
        initial_state: ReviewState = {
            'messages': [{"role": "user", "content": query}],
            'context': {},
            'intent': 'review_query',
            'reviews': None,
            'sentiment_filter': None,
            'product_name': None
        }
        
        final_state = self.graph.invoke(initial_state)

        return [
            msg.content
            for msg in final_state['messages'] 
            if isinstance(msg, AIMessage)
        ][0]

class Order:
    def __init__(self, product_name, customer_name, order_id=None, order_date=None, status=None, shipping_address=None, shipping_method=None):
        self.product_name = product_name
        self.customer_name = customer_name
        self.order_id = order_id or str(uuid.uuid4()) 
        self.order_date = datetime.now() if not order_date else datetime.fromisoformat(order_date)
        self.status = "Pending" if not status else status 
        self.shipping_address = None if not shipping_address else None
        self.shipping_method = None if not shipping_method else None

    def __str__(self):
        return f"Order ID: {self.order_id}, Product: {self.product_name}, Status: {self.status}"

    def to_dict(self):
        """Converts Order object to a dictionary for JSON serialization."""
        return {
            "order_id": self.order_id,
            "product_name": self.product_name,
            "customer_name": self.customer_name,
            "order_date": self.order_date.isoformat(),  # Convert datetime to ISO format
            "status": self.status,
            "shipping_address": self.shipping_address,
            "shipping_method": self.shipping_method
        }


from uuid import uuid4

class OrdersAgent:
    def __init__(self, db_session, order_orm, openai_api_key=None):
        self.db_session = db_session
        self.order_orm = order_orm
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize StateGraph
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create a StateGraph to handle order-related workflows."""
        graph = StateGraph(dict)

        # Define nodes
        graph.add_node("extract_intent", self._extract_intent)
        graph.add_node("process_query", self._process_order_query)

        # Define edges
        graph.add_edge(START, "extract_intent")
        graph.add_edge("extract_intent", "process_query")
        graph.add_edge("process_query", END)

        return graph.compile()

    def _extract_intent(self, state):
        messages = state['messages']
        query = messages[-1].content

        # Use OpenAI to classify intent
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify the order query intent as one of: CHECK_STATUS, PLACE_ORDER, UPDATE_STATUS, GENERAL_QUERY. Respond with only a single word."},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        intent = response.choices[0].message.content.strip()
        state['intent'] = intent
        return state

    def _process_order_query(self, state):
        intent = state['intent']
        query = state['messages'][-1].content

        if intent == "CHECK_STATUS":
            order_id = self._extract_order_id_from_query(query)
            if order_id:
                result = self.get_order_status(order_id)
            else:
                result = "Could not find order ID in query."

        elif intent == "PLACE_ORDER":
            # Extract product and customer from query using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract product_name and customer_name from the query. Return a JSON string which has product_name and customer_name in its attributes."},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            extracted = json.loads(response.choices[0].message.content)
            result = self.place_order(extracted['product_name'], extracted['customer_name'])

        elif intent == "UPDATE_STATUS":
            order_id = self._extract_order_id_from_query(query)
            # Extract new status from query
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract the new status from the query."},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            new_status = response.choices[0].message.content.strip()
            result = self.update_order_status(order_id, new_status)

        else:
            result = "I'm not sure how to help with that query."

        state['messages'].append(AIMessage(content=str(result)))
        return state

    def _extract_order_id_from_query(self, query: str) -> str:
        """Extracts order ID from query using more robust parsing."""
        try:
            # Use OpenAI to extract order ID
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract only the order ID from the query. Return just the ID."},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def place_order(self, product_name: str, customer_name: str) -> Dict:
        """Creates a new order in the database."""
        new_order = self.order_orm(
            id=str(uuid4()),
            product_name=product_name,
            customer_name=customer_name,
            status="PENDING"
        )

        self.db_session.add(new_order)
        self.db_session.commit()

        return f"Order placed successfully. Your order ID is: {new_order.id}"

    def get_order_status(self, order_id: str) -> str:
        """Retrieves order status from database."""
        order = self.db_session.query(self.order_orm).filter_by(id=order_id).first()

        if order:
            return f"Order ID: {order.id}, Status: {order.status}"
        else:
            return f"Order ID: {order_id} not found."

    def update_order_status(self, order_id: str, new_status: str) -> str:
        """Updates order status in database."""
        order = self.db_session.query(self.order_orm).filter_by(id=order_id).first()

        if order:
            order.status = new_status
            self.db_session.commit()
            return f"Order ID: {order_id} status updated to: {new_status}"
        else:
            return f"Order ID: {order_id} not found."

    def process_query(self, query: str) -> str:
        """Process natural language queries about orders using StateGraph."""
        initial_state = {
            "messages": [AIMessage(content=query)],
            "intent": None,
            "context": {}
        }

        final_state = self.graph.invoke(initial_state)

        return [
            msg.content 
            for msg in final_state['messages']
            if isinstance(msg, AIMessage)
        ][-1]


class MetaAgent:
    def __init__(self, openai_api_key, product_embeddings, order_orm, db_session):
        """Initialize individual agents and a meta-processing graph."""
        self.openai_api_key = openai_api_key
        self.product_embeddings = product_embeddings
        self.order_orm = order_orm
        self.db_session = db_session
        self.review_agent = self._initialize_review_agent()
        self.order_agent = self._initialize_order_agent()
        self.support_agent = self._initialize_product_details_agent()
        self.graph = self._build_meta_graph()
        self.openai_client = OpenAI(api_key=openai_api_key)

    def _initialize_review_agent(self):
        """Initialize the Review Agent."""
        return ProductReviewsAgent(openai_api_key=self.openai_api_key, db_session=self.db_session)  # Replace with the actual initialization of your Review Agent

    def _initialize_order_agent(self):
        """Initialize the Orders Agent."""
        return OrdersAgent(openai_api_key=self.openai_api_key, order_orm=self.order_orm, db_session=self.db_session)  # Replace with the actual initialization of your Orders Agent

    def _initialize_product_details_agent(self):
        """Initialize the Support Agent."""
        return ProductDetailsAgent(openai_api_key=self.openai_api_key, embedding_file=self.product_embeddings)  # Replace with the actual initialization of your Support Agent

    def _build_meta_graph(self):
        """Build the meta-processing graph."""
        graph_builder = StateGraph(dict)

        # Add nodes
        graph_builder.add_node("ParseIntent", self._parse_intent)
        graph_builder.add_node("DelegateToAgent", self._delegate_to_agent)
        graph_builder.add_node("GenerateResponse", self._generate_meta_response)

        # Add edges
        graph_builder.add_edge(START, "ParseIntent")
        graph_builder.add_edge("ParseIntent", "DelegateToAgent")
        graph_builder.add_edge("DelegateToAgent", "GenerateResponse")
        graph_builder.add_edge("GenerateResponse", END)

        return graph_builder.compile()

    def _parse_intent(self, state: dict) -> dict:
        """Parse the user query to identify intent and decide the agent."""

        response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=state['messages'],
                    temperature=0
                )
        response_text = response.choices[0].message.content

        if "review" in response_text.lower():
            state['intent'] = 'review'
        elif "order" in response_text.lower():
            state['intent'] = 'order'
        elif "product_details" in response_text.lower():
            state['intent'] = 'product_details'
        else:
            state['intent'] = 'unknown'
        return state

    def _delegate_to_agent(self, state: dict) -> dict:
        """Delegate the query to the appropriate agent based on intent."""
        intent = state['intent']
        query = state['messages'][-1]['content']

        if intent == 'review':
            responses = self.review_agent.process_query(query)
        elif intent == 'order':
            responses = self.order_agent.process_query(query)
        elif intent == 'product_details':
            responses = self.support_agent.process_query(query)
        else:
            responses = "I'm sorry, I couldn't understand your request."

        state['context']['responses'] = responses
        return state

    def _generate_meta_response(self, state: dict) -> dict:
        """Generate a unified response from the delegated agent's result."""
        responses = state['context'].get('responses', [])
        state['messages'].append({
            "role": "assistant",
            "content": "\n".join(responses)
        })
        return state

    def process_query(self, query: str) -> List[str]:
        """Process a query using the meta-agent."""
        initial_state = {
            'messages': [
                {"role": "user", "content": f"""
                 Your job is to identify the intent of the given query into 3 intents: review, order and product_details.
                 You only have to respond with one intent.
                 Only respond with one word.
                 If you cannot identify the intent with confidence reply unknown.
                 
                 Following are the responsibilities of each intent
                 1. order:
                    To place an order, to check the status of an order
                 2. product_details:
                    To get details of the product, compare multiple products
                 3. review
                    To get reviews of a product
                 
                 """},
                {"role": "user", "content": query}
            ],
            'context': {}
        }

        final_state = self.graph.invoke(initial_state)
        return final_state['context']['responses']