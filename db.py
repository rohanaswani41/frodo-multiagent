from sqlalchemy import create_engine, Column, Integer, String, Text, Date, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a base class for ORM models
Base = declarative_base()

# Define the Orders table as a Python class
class Order(Base):
    __tablename__ = 'orders'

    id = Column(Text, primary_key=True)
    product_name = Column(Text, nullable=False)
    customer_name = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)

    def __repr__(self):
        return f"<Order(id={self.id}, product_name='{self.product_name}', customer_name='{self.customer_name}', status='{self.status}'>"

class Product(Base):
    __tablename__ = 'product'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)

class ProductReview(Base):
    __tablename__ = 'product_review'
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey('product.id'), nullable=False)
    review_text = Column(Text, nullable=False)
    reviewer = Column(Text, nullable=False)
    rating = Column(Float, nullable=False)


# Create an in-memory SQLite database and session
engine = create_engine("sqlite:///:memory:")
Session = sessionmaker(bind=engine)
session = Session()

# Create the tables
Base.metadata.create_all(engine)


def load_products_data(data):
    for product_data in data["reviews"]:
        product_name = product_data["name"]
        
        # Add product to the database
        product = Product(name=product_name)
        session.add(product)
        session.commit()  # Commit to generate the product ID
        
        # Add associated reviews
        for review_data in product_data["reviews"]:
            review = ProductReview(
                product_id=product.id,
                review_text=review_data["comment"],
                reviewer=review_data['reviewer'],
                rating=review_data['rating']
            )
            session.add(review)
    
    # Commit all reviews
    session.commit()
# Example Usage
# if __name__ == "__main__":
#     # Add new orders
#     order1 = Order(product_name="Laptop", quantity=1, customer_name="Alice", status="Processing", delivery_date="2025-01-20")
#     order2 = Order(product_name="Phone", quantity=2, customer_name="Bob", status="Shipped", delivery_date="2025-01-18")
#     session.add(order1)
#     session.add(order2)
#     session.commit()

#     # Retrieve all orders
#     print("All Orders:")
#     for order in session.query(Order).all():
#         print(order)

#     # Get an order by ID
#     print("\nOrder with ID 1:")
#     print(session.query(Order).filter_by(id=1).first())

#     # Update an order's status
#     order_to_update = session.query(Order).filter_by(id=1).first()
#     order_to_update.status = "Delivered"
#     session.commit()
#     print("\nUpdated Order with ID 1:")
#     print(session.query(Order).filter_by(id=1).first())

#     # Delete an order
#     session.query(Order).filter_by(id=2).delete()
#     session.commit()
#     print("\nAll Orders after deleting ID 2:")
#     for order in session.query(Order).all():
#         print(order)
