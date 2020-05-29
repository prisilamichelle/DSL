from bs4 import BeautifulSoup
import base64
import json

input_file_name = input("Please enter the input file name : ")
text_file = open(input_file_name, 'r')
crawled_data = text_file.readlines()
text_file.close()

html_docs = []
for line in crawled_data :
    values = json.loads(line)
    if values['relevance']['isRelevant']:
        html_docs.append(values['content'])

all_text = []
print(len(html_docs))
for code in html_docs:
    html_doc = base64.b64decode(code).decode(errors='replace')
    # print(html_doc)

    soup = BeautifulSoup(html_doc, 'html.parser')

    all_p = soup.find_all('p')
    p_list = []
    for p in all_p:
        text = p.get_text() + '\n'
        p_list.append(text)
        # print(p.get_text())
    all_text.append(p_list)

print('done')

output_file_name = input("Please enter the output file name : ")
with open(output_file_name,"w") as f:
    for txt in all_text:
        for p in txt:
            f.write(p)
