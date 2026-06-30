from ai_student_impact.rag import RagChatbot

queries = [
    "paid subscription students",
    "students in Humanities",
    "records where Major_Category is Humanities",
    "Paid_Subscription",
    "Major_Category",
]

bot = RagChatbot(max_rows=100)
retr = bot.vector_store.as_retriever(search_kwargs={"k": 6})
print("=== RETRIEVAL DEBUG ===")
for q in queries:
    print("QUERY:", q)
    docs = retr.get_relevant_documents(q)
    for i, d in enumerate(docs, 1):
        print(f"DOC {i}: {d.metadata}")
        print(d.page_content[:400].replace('\n', ' '))
        print('---')
    print('=======================')
