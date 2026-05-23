from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, chain

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="qwen2.5-3b-instruct",
    temperature=0,
    max_completion_tokens=250
)

list_instructions = CommaSeparatedListOutputParser().get_format_instructions()
list_output_parser = CommaSeparatedListOutputParser()
prompt_template = ChatPromptTemplate.from_messages(['human',
                                                    "give me the list of countries in {continent} \n"
                                                    + list_instructions])

#template_result = prompt_template.invoke({'continent': 'middle east'})
#result = llm.invoke(template_result)
#response = list_output_parser.invoke(result)

#instead of using invokes use chain
chain = prompt_template | llm | list_output_parser
#print(chain.invoke({'continent': 'middle east'}))


prompt_template2 = ChatPromptTemplate.from_messages(['human',
                                                    "give me some information about {country} and {things} about it \n"])

chain2 = prompt_template2 | llm

#IPython magic commands %%time
response = chain2.batch([{'country': 'Iran', 'things': 'weather'},
                    {'country': 'USA', 'things': 'culture'}])
#print(response[0].content)

#stream, os good for used memory
respnse2 = chain2.stream({'country': 'Iran', 'things': 'weather'}) 
#for i in respnse2:
    #print(i.content, end='')

#chain is a RunnableSequence class, where the output of each is the input of the next
#ChatPromptTemplate start with Runnable class, therefore inheriting the invoke(), batch() and stream()
#other parsers and chattemplates are inheriting from a runnable or serialaing class
#Runnable: it is a unit of work that can be invoked, batched, streamed, transformed and composed.
#we can have a chain of chains
#search for a table to have langchain components and input-output types

#RunnablePassthrough, it is langchains identity function
RunnablePassthrough().invoke([1,2,3])

tools_prompt = ChatPromptTemplate.from_template('''what are the top five skills a person need to become a {job}''')
strategy_prompt = ChatPromptTemplate.from_template('''considering the tools provided, develop a strategy to leraning them: {tools}''')
string_parser = StrOutputParser()

chain_tool = tools_prompt | llm | string_parser | {'tools':RunnablePassthrough()}
chain_strategy = strategy_prompt | llm | string_parser
main_chain = chain_tool | chain_strategy
#print(main_chain.invoke({'job': 'AI Engineering'}))

#visualaze the chain --> pip install grandalf
#main_chain.get_graph().print_ascii()

#Runnable Parallel
template_books = ChatPromptTemplate.from_template('''suggest top three {programming} books. 
                                             answer only by listing the books.''')
template_projects = ChatPromptTemplate.from_template('''suggest three {programming} projects suitable for beginer programmers.
                                             answer only by listing the projects.''')
chain_books = template_books | llm | string_parser
chain_projects = template_projects | llm | string_parser
chain_parallel = RunnableParallel({'books': chain_books, 'projects': chain_projects})
#print(chain_parallel.invoke({'programming': 'rust'}))

template_time = ChatPromptTemplate.from_template('''I am a beginer. consider the following books: {books}
                                                 also, consider the following projects: {projects}
                                                 how much time would it take me to complete the books and projects?''')

chain_time = (  chain_parallel  #{'books': chain_books, 'projects': chain_projects} we can use it because parallel is default
              | template_time
              | llm
              | string_parser
              )
#print(chain_time.invoke({'programming': 'Rust'}))

#transform ordinary lambda functions to Runnable objects using Lambda
import pandas as pd
from nltk.stem import PorterStemmer
from langchain_core.runnables import RunnableLambda
ps = PorterStemmer()
lmbda = lambda x: ps.stem(x)
lmdba_u = lambda x: [x.upper()]


runnable_lmbda = RunnableLambda(lmbda)
runnable_lmbda_u = RunnableLambda(lmdba_u)
chain = runnable_lmbda | runnable_lmbda_u

print(chain.invoke('collection'))

#Decorator @chain, means a function modifies the behavior of another function or class, enhancing their functionality
@chain
def runnable_stem(x):
    return ps.stem(x)

print(type(runnable_stem))
print(runnable_stem.invoke('collection'))