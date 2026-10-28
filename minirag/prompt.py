from .schema import (
    DEFAULT_ENTITY_TYPES,
    render_entity_schema,
    render_relationship_schema,
)

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = DEFAULT_ENTITY_TYPES
PROMPTS["ENTITY_SCHEMA"] = render_entity_schema()
PROMPTS["RELATIONSHIP_SCHEMA"] = render_relationship_schema()


PROMPTS["entity_extraction"] = """-Goal-
Given a text document, identify every entity that belongs to the historical entity schema below, together with all relationships among those entities.

-Entity schema-
Every entity MUST use one of these types: [{entity_types}].
Base attributes (extract for EVERY entity):
- entity_name: Name of the entity, in the same language as the input text.
- entity_type: One of [{entity_types}]. Do NOT invent types outside this list.
- time_range: Temporal bounds associated with the entity, e.g. "1225-1258", "1428", "thế kỷ XIII". Use "" if unknown.
- description: Comprehensive description of the entity's attributes and activities.
- alternative_names: Comma-separated other names/aliases that refer to the SAME entity. Always include entity_name itself. Do NOT include pronouns such as "he", "she", "they", "it". Use "" if there are no other names.
- geo_location: Geographical coordinates or place names associated with the entity. Use "" if unknown.
Type-specific attributes (extract ONLY for the matching entity_type; use "" or omit the key when unknown):
{entity_schema}

-Output format-
1. For each identified entity, output:
("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<time_range>"{tuple_delimiter}"<description>"{tuple_delimiter}"<alternative_names>"{tuple_delimiter}<attributes_json>)
- <attributes_json> is a compact JSON object containing ONLY the type-specific attributes valid for that entity_type, plus geo_location when known. Use {{}} when there are none.
- Never put entity_name, entity_type, description, time_range or alternative_names inside <attributes_json>.
- Output valid JSON with double quotes; omit keys whose value is unknown.

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
{relationship_schema}
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [PERSON, PLACE, TIME, EVENT, ARTIFACT, OBJECT, CONCEPT]
Text:
Năm 1428, sau khi đánh tan quân Minh, Lê Lợi lên ngôi Hoàng đế tại Đông Kinh, mở đầu triều Hậu Lê. Cuộc khởi nghĩa Lam Sơn (1418-1427) do ông lãnh đạo đã chấm dứt hai mươi năm đô hộ. Vào thế kỷ XV, Đông Kinh trở thành trung tâm chính trị của Đại Việt.
################
Output:
("entity"{tuple_delimiter}"Lê Lợi"{tuple_delimiter}"PERSON"{tuple_delimiter}"1385-1433"{tuple_delimiter}"Lê Lợi là người lãnh đạo khởi nghĩa Lam Sơn và lên ngôi năm 1428, mở đầu triều Hậu Lê."{tuple_delimiter}"Lê Lợi, Lê Thái Tổ"{tuple_delimiter}{{"birth_date": "1385", "death_date": "1433", "geo_location": "Lam Sơn"}}){record_delimiter}
("entity"{tuple_delimiter}"Đông Kinh"{tuple_delimiter}"PLACE"{tuple_delimiter}""{tuple_delimiter}"Đông Kinh là kinh đô của Đại Việt, được xây dựng lại sau khởi nghĩa Lam Sơn."{tuple_delimiter}"Đông Kinh, Thăng Long"{tuple_delimiter}{{"location_type": "kinh đô"}}){record_delimiter}
("entity"{tuple_delimiter}"Khởi nghĩa Lam Sơn"{tuple_delimiter}"EVENT"{tuple_delimiter}"1418-1427"{tuple_delimiter}"Khởi nghĩa Lam Sơn là cuộc chiến do Lê Lợi lãnh đạo nhằm chấm dứt ách đô hộ."{tuple_delimiter}"Khởi nghĩa Lam Sơn"{tuple_delimiter}{{"start_time": "1418", "end_time": "1427", "location": "Lam Sơn"}}){record_delimiter}
("entity"{tuple_delimiter}"Thế kỷ XV"{tuple_delimiter}"TIME"{tuple_delimiter}"1401-1500"{tuple_delimiter}"Thế kỷ XV là bối cảnh diễn ra khởi nghĩa Lam Sơn và triều Hậu Lê."{tuple_delimiter}"Thế kỷ XV"{tuple_delimiter}{{"temporal_value": "1401-1500"}}){record_delimiter}
("entity"{tuple_delimiter}"Lam Sơn"{tuple_delimiter}"PLACE"{tuple_delimiter}""{tuple_delimiter}"Lam Sơn là căn cứ địa của nghĩa quân Lê Lợi."{tuple_delimiter}"Lam Sơn"{tuple_delimiter}{{"location_type": "căn cứ địa"}}){record_delimiter}
("relationship"{tuple_delimiter}"Lê Lợi"{tuple_delimiter}"Khởi nghĩa Lam Sơn"{tuple_delimiter}"Lê Lợi là người lãnh đạo cuộc khởi nghĩa Lam Sơn."{tuple_delimiter}"lãnh đạo, khởi nghĩa"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Khởi nghĩa Lam Sơn"{tuple_delimiter}"Đông Kinh"{tuple_delimiter}"Sau thắng lợi của khởi nghĩa, Đông Kinh trở thành kinh đô của triều Hậu Lê."{tuple_delimiter}"kinh đô, chính trị"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"khởi nghĩa Lam Sơn, triều Hậu Lê, Đại Việt, thế kỷ XV"){completion_delimiter}
#############################

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""


PROMPTS[
    "entiti_continue_extraction_mini"
] = """MANY entities were missed in the last extraction.
After summarizing with all the information previously extracted, compared to the original text, it was noticed that the following information was mainly omitted:
{omit}

Use ONLY the entity types and attribute schema given in the original instructions.
Do not invent new entity types.

Entity_types: {entity_types}


Add them below using the same format, including the final <attributes_json> field:
"""


PROMPTS["minirag_query2kwd"] = """---Role---

You are a helpful assistant tasked with identifying both answer-type and low-level keywords in the user's query.

---Goal---

Given the query, list both answer-type and low-level keywords.
answer_type_keywords focus on the type of the answer to the certain query, while low-level keywords focus on specific entities, details, or concrete terms.
The answer_type_keywords must be selected from Answer type pool.
This pool is in the form of a dictionary, where the key represents the Type you should choose from and the value represents the example samples.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have three keys:
  - "answer_type_keywords" for the types of the answer. In this list, the types with the highest likelihood should be placed at the forefront. No more than 3.
  - "entities_from_query" for specific entities or details. It must be extracted from the query.
######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
Answer type pool: {{
 'PERSONAL LIFE': ['FAMILY TIME', 'HOME MAINTENANCE'],
 'STRATEGY': ['MARKETING PLAN', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['JANE DOE', 'JOHN SMITH'],
 'FOOD': ['PASTA', 'SUSHI'],
 'EMOTION': ['HAPPINESS', 'ANGER'],
 'PERSONAL EXPERIENCE': ['TRAVEL ABROAD', 'STUDYING ABROAD'],
 'INTERACTION': ['TEAM MEETING', 'NETWORKING EVENT'],
 'BEVERAGE': ['COFFEE', 'TEA'],
 'PLAN': ['ANNUAL BUDGET', 'PROJECT TIMELINE'],
 'GEO': ['NEW YORK CITY', 'SOUTH AFRICA'],
 'GEAR': ['CAMPING TENT', 'CYCLING HELMET'],
 'EMOJI': ['🎉', '🚀'],
 'BEHAVIOR': ['POSITIVE FEEDBACK', 'NEGATIVE CRITICISM'],
 'TONE': ['FORMAL', 'INFORMAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["STRATEGY","PERSONAL LIFE"],
  "entities_from_query": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "When was SpaceX's first rocket launch?"
Answer type pool: {{
 'DATE AND TIME': ['2023-10-10 10:00', 'THIS AFTERNOON'],
 'ORGANIZATION': ['GLOBAL INITIATIVES CORPORATION', 'LOCAL COMMUNITY CENTER'],
 'PERSONAL LIFE': ['DAILY EXERCISE ROUTINE', 'FAMILY VACATION PLANNING'],
 'STRATEGY': ['NEW PRODUCT LAUNCH', 'YEAR-END SALES BOOST'],
 'SERVICE FACILITATION': ['REMOTE IT SUPPORT', 'ON-SITE TRAINING SESSIONS'],
 'PERSON': ['ALEXANDER HAMILTON', 'MARIA CURIE'],
 'FOOD': ['GRILLED SALMON', 'VEGETARIAN BURRITO'],
 'EMOTION': ['EXCITEMENT', 'DISAPPOINTMENT'],
 'PERSONAL EXPERIENCE': ['BIRTHDAY CELEBRATION', 'FIRST MARATHON'],
 'INTERACTION': ['OFFICE WATER COOLER CHAT', 'ONLINE FORUM DEBATE'],
 'BEVERAGE': ['ICED COFFEE', 'GREEN SMOOTHIE'],
 'PLAN': ['WEEKLY MEETING SCHEDULE', 'MONTHLY BUDGET OVERVIEW'],
 'GEO': ['MOUNT EVEREST BASE CAMP', 'THE GREAT BARRIER REEF'],
 'GEAR': ['PROFESSIONAL CAMERA EQUIPMENT', 'OUTDOOR HIKING GEAR'],
 'EMOJI': ['📅', '⏰'],
 'BEHAVIOR': ['PUNCTUALITY', 'HONESTY'],
 'TONE': ['CONFIDENTIAL', 'SATIRICAL'],
 'LOCATION': ['CENTRAL PARK', 'DOWNTOWN LIBRARY']
}}

################
Output:
{{
  "answer_type_keywords": ["DATE AND TIME", "ORGANIZATION", "PLAN"],
  "entities_from_query": ["SpaceX", "Rocket launch", "Aerospace", "Power Recovery"]

}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
Answer type pool: {{
 'PERSONAL LIFE': ['MANAGING WORK-LIFE BALANCE', 'HOME IMPROVEMENT PROJECTS'],
 'STRATEGY': ['MARKETING STRATEGIES FOR Q4', 'EXPANDING INTO NEW MARKETS'],
 'SERVICE FACILITATION': ['CUSTOMER SATISFACTION SURVEYS', 'STAFF RETENTION PROGRAMS'],
 'PERSON': ['ALBERT EINSTEIN', 'MARIA CALLAS'],
 'FOOD': ['PAN-FRIED STEAK', 'POACHED EGGS'],
 'EMOTION': ['OVERWHELM', 'CONTENTMENT'],
 'PERSONAL EXPERIENCE': ['LIVING ABROAD', 'STARTING A NEW JOB'],
 'INTERACTION': ['SOCIAL MEDIA ENGAGEMENT', 'PUBLIC SPEAKING'],
 'BEVERAGE': ['CAPPUCCINO', 'MATCHA LATTE'],
 'PLAN': ['ANNUAL FITNESS GOALS', 'QUARTERLY BUSINESS REVIEW'],
 'GEO': ['THE AMAZON RAINFOREST', 'THE GRAND CANYON'],
 'GEAR': ['SURFING ESSENTIALS', 'CYCLING ACCESSORIES'],
 'EMOJI': ['💻', '📱'],
 'BEHAVIOR': ['TEAMWORK', 'LEADERSHIP'],
 'TONE': ['FORMAL MEETING', 'CASUAL CONVERSATION'],
 'LOCATION': ['URBAN CITY CENTER', 'RURAL COUNTRYSIDE']
}}

################
Output:
{{
  "answer_type_keywords": ["STRATEGY", "PERSON"],
  "entities_from_query": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
Example 4:

Query: "Where is the capital of the United States?"
Answer type pool: {{
 'ORGANIZATION': ['GREENPEACE', 'RED CROSS'],
 'PERSONAL LIFE': ['DAILY WORKOUT', 'HOME COOKING'],
 'STRATEGY': ['FINANCIAL INVESTMENT', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['ALBERTA SMITH', 'BENJAMIN JONES'],
 'FOOD': ['PASTA CARBONARA', 'SUSHI PLATTER'],
 'EMOTION': ['HAPPINESS', 'SADNESS'],
 'PERSONAL EXPERIENCE': ['TRAVEL ADVENTURE', 'BOOK CLUB'],
 'INTERACTION': ['TEAM BUILDING', 'NETWORKING MEETUP'],
 'BEVERAGE': ['LATTE', 'GREEN TEA'],
 'PLAN': ['WEIGHT LOSS', 'CAREER DEVELOPMENT'],
 'GEO': ['PARIS', 'NEW YORK'],
 'GEAR': ['CAMERA', 'HEADPHONES'],
 'EMOJI': ['🏢', '🌍'],
 'BEHAVIOR': ['POSITIVE THINKING', 'STRESS MANAGEMENT'],
 'TONE': ['FRIENDLY', 'PROFESSIONAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["LOCATION"],
  "entities_from_query": ["capital of the United States", "Washington", "New York"]
}}
#############################

-Real Data-
######################
Query: {query}
Answer type pool:{TYPE_POOL}
######################
Output:

"""


PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
