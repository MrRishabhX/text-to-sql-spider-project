# import streamlit as st
# st.title("Text Box and Button Example")
# #Create the first text input box
# text_input_1 = st.text_input("Enter text for Box 1:")


# #Create a button
# if st.button("Process Text"):
# # This code block executes when the button is clicked

#     st.write(f"Text from Box 1: {text_input_1}")
# # model fetch code in the last cell of colab notebook
# # ab db connection code yha par fir sql query execute hogi
# # populate the result in tabular format
    
# #Add any further processing or logic here





import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Text-to-SQL with MySQL Spider DB",
    page_icon="🔍",
    layout="wide"
)

# Database configuration - UPDATE THESE WITH YOUR MYSQL CREDENTIALS
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',    # Your MySQL username
    'password': 'root', # Your MySQL password  
    'database': 'spider_db'           # The database we just created
}

# Model path - UPDATE THIS TO YOUR LOCAL MODEL PATH
MODEL_PATH = r"C:\Users\hp\OneDrive\Desktop\Project\t5-finetuned-rag-compact"  # Path to your downloaded model

@st.cache_resource
def load_model():
    """Load the fine-tuned T5 model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def setup_retrieval_system():
    """Setup retrieval system with MySQL schema"""
    try:
        # Connect to MySQL and extract schema
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        retrieval_documents = []
        
        # Get all tables
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        for table in tables:
            # Get column information
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            col_desc = ", ".join([f"{col[0]} ({col[1]})" for col in columns])
            retrieval_documents.append(f"Table: {table}. Columns: {col_desc}.")
            
            # Get foreign key information
            cursor.execute(f"""
            SELECT 
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = '{DB_CONFIG["database"]}'
            AND TABLE_NAME = '{table}'
            AND REFERENCED_TABLE_NAME IS NOT NULL
            """)
            
            fks = cursor.fetchall()
            for fk in fks:
                retrieval_documents.append(
                    f"Join: {table}.{fk[0]} -> {fk[1]}.{fk[2]}"
                )
        
        connection.close()
        
        # Create embeddings
        embedder = SentenceTransformer("all-mpnet-base-v2")
        doc_embeddings = embedder.encode(retrieval_documents, convert_to_numpy=True)
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        
        return embedder, index, retrieval_documents
        
    except Exception as e:
        st.error(f"Error setting up retrieval system: {e}")
        return None, None, None

def retrieve_schema(query, embedder, index, retrieval_documents, k=3):
    """Retrieve relevant schema for the query"""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_emb, k)
    return [retrieval_documents[i] for i in indices[0]]

def generate_sql(question, model, tokenizer, embedder, index, retrieval_documents, device):
    """Generate SQL from natural language"""
    try:
        # Get relevant schema
        schema_context = retrieve_schema(question, embedder, index, retrieval_documents, k=3)
        schema_text = " ; ".join(schema_context)
        
        # Create prompt
        prompt = f"translate English to SQL: Question: {question.strip()} Schema: {schema_text} SQL:"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=180,
                num_beams=6,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                length_penalty=1.2
            )
        
        sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sql_query, schema_text
        
    except Exception as e:
        return f"Error: {e}", ""

def execute_mysql_query(sql_query):
    """Execute SQL query on MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        cursor.execute(sql_query)
        
        if sql_query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            connection.close()
            return {'success': True, 'data': results, 'columns': columns}
        else:
            connection.commit()
            affected_rows = cursor.rowcount
            connection.close()
            return {'success': True, 'affected_rows': affected_rows}
            
    except Error as e:
        return {'success': False, 'error': str(e)}

def main():
    st.title("🔍 Text-to-SQL Generator with MySQL Spider Database")
    st.markdown("*Convert natural language to SQL and execute on your Spider MySQL database*")
    
    # Load model and setup retrieval
    with st.spinner("🔄 Loading model and connecting to database..."):
        model, tokenizer, device = load_model()
        embedder, index, retrieval_documents = setup_retrieval_system()
    
    if model is None or embedder is None:
        st.error("❌ Failed to initialize system. Please check your setup.")
        return
    
    st.success("✅ System ready! Model loaded and connected to MySQL Spider database.")
    
    # Sidebar with database info
    with st.sidebar:
        st.header("🗄️ Database Info")
        
        # Show tables
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            st.write("**Available Tables:**")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                st.write(f"• {table} ({count} rows)")
            
            connection.close()
            
        except Error as e:
            st.error(f"Database connection error: {e}")
        
        st.markdown("---")
        st.header("📝 Example Queries")
        examples = [
            "Show all Computer Science students with GPA above 3.7",
            "Count students by major", 
            "Find employees with salary greater than 70000",
            "List all courses in Computer Science department",
            "Show average salary by department",
            "Find all orders placed in January 2024",
            "Show students enrolled in Database Systems course",
            "List books published after 1950"
        ]
        
        for example in examples:
            if st.button(f"📋 {example}", key=f"ex_{hash(example)}"):
                st.session_state.example_query = example
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Handle example selection
        default_query = ""
        if 'example_query' in st.session_state:
            default_query = st.session_state.example_query
            del st.session_state.example_query
        
        user_question = st.text_area(
            "Enter your question:",
            value=default_query,
            height=100,
            placeholder="e.g., Show me all students majoring in Computer Science with GPA above 3.5"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            generate_btn = st.button("🚀 Generate SQL", type="primary")
        with col_b:
            execute_btn = st.button("▶️ Generate & Execute", type="secondary")
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        if st.button("🔍 Show Database Schema"):
            st.session_state.show_schema = True
        
        if st.button("📊 Show Sample Data"):
            st.session_state.show_samples = True
    
    # Process query
    if (generate_btn or execute_btn) and user_question.strip():
        with st.spinner("🧠 Generating SQL..."):
            sql_result, schema_context = generate_sql(
                user_question, model, tokenizer, embedder, index, retrieval_documents, device
            )
        
        # Display generated SQL
        st.markdown("### 📄 Generated SQL")
        st.code(sql_result, language="sql")
        
        # Show retrieved schema context
        with st.expander("🔍 Retrieved Schema Context"):
            for i, context in enumerate(schema_context.split(" ; "), 1):
                if context.strip():
                    st.write(f"{i}. {context}")
        
        # Execute if requested
        if execute_btn:
            st.markdown("### 📊 Query Results")
            with st.spinner("⚡ Executing on MySQL database..."):
                result = execute_mysql_query(sql_result)
            
            if result['success']:
                if 'data' in result and result['data']:
                    df = pd.DataFrame(result['data'], columns=result['columns'])
                    st.dataframe(df, use_container_width=True)
                    st.success(f"✅ Retrieved {len(result['data'])} rows")
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        csv,
                        "query_results.csv",
                        "text/csv"
                    )
                elif 'affected_rows' in result:
                    st.success(f"✅ Query executed! {result['affected_rows']} rows affected")
                else:
                    st.info("ℹ️ Query executed successfully (no results)")
            else:
                st.error(f"❌ Query failed: {result['error']}")
    
    # Show schema if requested
    if st.session_state.get('show_schema', False):
        st.markdown("### 🗂️ Database Schema")
        for doc in retrieval_documents:
            st.write(f"• {doc}")
        st.session_state.show_schema = False
    
    # Show sample data if requested
    if st.session_state.get('show_samples', False):
        st.markdown("### 📋 Sample Data")
        try:
            connection = mysql.connector.connect(**DB_CONFIG)
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            for table in tables[:4]:  # Show first 4 tables
                st.write(f"**{table}:**")
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                data = cursor.fetchall()
                cursor.execute(f"DESCRIBE {table}")
                columns = [col[0] for col in cursor.fetchall()]
                
                if data:
                    df = pd.DataFrame(data, columns=columns)
                    st.dataframe(df)
                else:
                    st.write("No data")
            
            connection.close()
            
        except Error as e:
            st.error(f"Error fetching sample data: {e}")
            
        st.session_state.show_samples = False

if __name__ == "__main__":
    main()
