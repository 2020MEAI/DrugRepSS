import requests, datetime, json
import pandas as pd

def geteb (drug):

# 替换为 OpenAI API 密钥
    api_key = "sk-AisP0mRH7jpEDv97HixgA2Uc8w64Zwf8P1imbcHVbPSInJaL"

    # 构建请求头部
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建请求体
    data = {
        "input": drug,
        "model": "text-embedding-ada-002"
    }

    # 发送 POST 请求
    url = "https://api.openai-proxy.org/v1/embeddings"
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        new_em = result['data'][0]['embedding']
        # 输出语义向量
        # print(result['data'][0]['embedding'])
        # print(f"***{len(result['data'][0]['embedding'])}***")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        exit(0)
    return new_em

def geteb(disease):
    # 替换为 ChatMindAi API 密钥
    api_key = "sk-AisP0mRH7jpEDv97HixgA2Uc8w64Zwf8P1imbcHVbPSInJaL"
    # 构建请求头部
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # 构建请求体
    data = {
        "input": disease,
        "model": "text-embedding-ada-002"
    }
    # 发送 POST 请求
    url = "https://api.ChatMindAi-proxy.org/v1/embeddings"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 检查响应状态码
    if response.status_code == 200:
        result = response.json()
        new_em = result['data'][0]['embedding']
        # 输出语义向量
        # print(result['data'][0]['embedding'])
        # print(f"***{len(result['data'][0]['embedding'])}***")
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        exit(0)
    return new_em

def readDrugName():
    drug_id = {}
    id = []
    # 指定 Excel 文件的路径
    excel_file = ".\\dataset\\drug_id.xlsx"

    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(excel_file)

    # 遍历 Excel 表格的每一行，将第二列数据作为键，第一列数据作为值存入字典
    for index, row in df.iterrows():
        drug_id[row[1]] = row[0]  

    return drug_id

def readDiseaseName():
    disease_id = {}
    # 指定 Excel 文件的路径
    excel_file = ".\\dataset\\disease_id.xlsx"
    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(excel_file)
    # 遍历 Excel 表格的每一行，将第二列数据作为键，第一列数据作为值存入字典
    for index, row in df.iterrows():
        disease_id[row[1]] = row[0]
    return disease_id

df = pd.read_excel(".\\dataset\\drug_id.xls")

drug_em = {}
print(f"在{datetime.datetime.now()}开始生成药物向量")
for index, row in df.iterrows():
    drug_em[row[1]] = geteb(row[0])
    df.at[index, "Drug_Vector"] = str(drug_em[row[1]])
    # print(len(drug_em[row[1]]))

df.to_excel(".\\dataset\\drug_id_vector.xlsx", index = False)
print(f"药物向量在{datetime.datetime.now()}生成完毕")

df = pd.read_excel(".\\dataset\\disease_id.xlsx")
disease_em = {}
print(f"在{datetime.datetime.now()}开始生成疾病向量")
for index, row in df.iterrows():
    disease_em[row[1]] = geteb(row[0])
    df.at[index, "Disease_Vector"] = str(disease_em[row[1]])
# print(len(disease_em[row[1]]))
df.to_excel(".\\dataset\\disease_id_vector.xlsx", index = False)
print(f"疾病向量在{datetime.datetime.now()}生成完毕")