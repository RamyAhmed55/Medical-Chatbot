# Any Instructions can u change it to make model follow specific rules

system_prompt = (
    "Use the following pieces of retrieved context to answer the user's question. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)