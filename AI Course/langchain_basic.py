from langchain_openai.chat_models import ChatOpenAI 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (PromptTemplate, 
                                    FewShotChatMessagePromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    AIMessagePromptTemplate)
from langchain_core.output_parsers import (StrOutputParser,
                                           CommaSeparatedListOutputParser)
#from langchain.output_parsers import DateTimeOutputParser

llm = ChatOpenAI(
    base_url="http://localhost:1234/api/v1",
    api_key="not-needed",
    model="google/gemma-3-4b",
    temperature=0,
    max_completion_tokens=250
)

#response = llm.invoke('give me some information about Data Analytics')
Human_message = HumanMessage(content='''I want to analysis the sales data to know some patterns for deicion making.
                                        I have some information about sale factors, cutomers and products.
                                        how can I predict the sales amount for next 2 month? ''')

System_message = SystemMessage(content='''your name is DataAnalyst. you analysis data as a professional.
                               you do it with a lot of exitement. you always suprise everyone with your results.''')

AI_message = AIMessage(content='''in this month, the sales amount increases by 70 percent in comprison with last year's current month.
                       but the number of product for sales didn't change in comparison with last year. it is shows that
                       inflation exists. the data  shows that 25 percent of customers have not ordered the products in last 3 month.
                       most of them are located in one region the shows some companies are seizing the market.''')

#response = llm.invoke([Human_message, System_message, AI_message])
#print(response.content)

Template = '''
system: {description}
human: I want to have informatin about {things}
'''
prompt_template = PromptTemplate.from_template(template=Template)
prompt_value = prompt_template.invoke({
    'description': '''the chatbot should answer with rationality and excitement''',
    'things': 'universe'
})

Template_system = '{description}'
Template_human = '''I want to have informatin about {things}'''
Sys_PromptTemplate = SystemMessagePromptTemplate.from_template(template=Template_system)
H_PromptTemplate = HumanMessagePromptTemplate.from_template(template=Template_human)
chat_template = ChatPromptTemplate.from_messages([Sys_PromptTemplate,H_PromptTemplate])
chat_value = chat_template.invoke({'description': '''the chatbot should answer with rationality and excitement''',
                                   'things': 'universe'})
#response = llm.invoke(chat_value)
#print(response.content)

AI_PromptTemplate = AIMessagePromptTemplate.from_template(template=Template_system)
chat_template2 = ChatPromptTemplate.from_messages([H_PromptTemplate, AI_PromptTemplate])
example = [{
            'things': 'running',
            'description': 'runnig is an exercise that make your heart muscles more powerful and improve your overal health'
            },
            {'things': 'data engineering',
             'description': 'data engineering is a job that has a important role in data aggregation and a person as a data engineer usually know various technology trends'
            }]
fewshot_prompttemplate = FewShotChatMessagePromptTemplate(examples= example,
                                                          example_prompt= chat_template2,
                                                          input_variables= ['things'])
chat_template_fewshot = ChatPromptTemplate.from_messages([fewshot_prompttemplate, H_PromptTemplate])
chat_value2 = chat_template_fewshot.invoke({'things': 'leadership'})
#for i in chat_value2.messages:
#    print(f'{i.type}: {i.content}\n')

#response = llm.invoke(chat_value2)
#print(chat_value2)
#print(response.content)


human_str = HumanMessage(content='give me the list of countries in middle east')
response_str = llm.invoke([human_str])
str = StrOutputParser()
response_parser_str = str.invoke(response_str)
print(response_parser_str)

human_list = HumanMessage(content=f'''give me the list of countries in middle east
                          {CommaSeparatedListOutputParser().get_format_instructions()}
                          ''')
response_list = llm.invoke([human_list])
list = CommaSeparatedListOutputParser()
response_parser_list = list.invoke(response_list)
print(response_parser_list)