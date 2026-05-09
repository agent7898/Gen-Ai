import fitz  # PyMuPDF


def extract(file):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def search(query, ipc):
    """Search for a query in the IPC document."""
    query = query.lower()
    lines = ipc.split("\n")
    results = [line for line in lines if query in line.lower()]
    
    if results:
        return results[:15]
    else:
        return ["No relevant section found."]


# Step 3: Main Chatbot Function
def chatbot():
    """Interactive chatbot for IPC document queries."""
    print("Loading IPC document...")
    ipc = extract(r"/Users/ananth/Desktop/ipc.pdf")
    
    while True:
        query = input("Ask a question about the IPC: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        results = search(query, ipc)
        print("\n".join(results))
        print("-" * 50)


chatbot()
