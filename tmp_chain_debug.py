from ai_student_impact.rag import RagChatbot

queries = [
    "paid subscription students",
    "students in Humanities",
    "records where Major_Category is Humanities",
]

bot = RagChatbot(max_rows=100)
for q in queries:
    print("QUERY:", q)
    result = bot.ask(q)
    print("ANSWER:", result['answer'])
    print("SOURCES:", result['sources'])
    print('---')
