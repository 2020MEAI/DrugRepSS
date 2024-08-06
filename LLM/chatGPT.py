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

def readDrugName():
    drug_id = {}
    id = []
    # 指定 Excel 文件的路径
    excel_file = "drug_id.xlsx"

    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(excel_file)

    # 遍历 Excel 表格的每一行，将第二列数据作为键，第一列数据作为值存入字典
    for index, row in df.iterrows():
        drug_id[row[1]] = row[0]  

    return drug_id

def addDrugNameAndVector(drugIdDict, drug_em):
    for sn in ["训练集", "测试集"]:
        df = pd.read_excel("训练集和测试集1V5.xlsx", sheet_name = sn)
        df["Drug_Name"] = None
        df["Drug_Vector"] = None
        print(f"在{datetime.datetime.now()}开始生成{sn}")
        for index, row in df.iterrows():
            if row.iloc[1] in drugIdDict.keys():
                add_name = drugIdDict[row.iloc[1]]
                df.at[index, "Drug_Name"] = add_name
        with open(f"{sn}output.txt", "w") as f:
            for index, row in df.iterrows():
                info = "\""
                if row.iloc[3] == "Drug_Name":
                    continue
                d_id = row.iloc[1]
                add_vector = drug_em[d_id]
                ef_rank = row.iloc[2]
                for i in range(len(add_vector)):
                    info += str(i + 1) + ':' + str(add_vector[i]) + ' '
                info += '\"'
                df.at[index, "Drug_Vector"] = info
                f.write(f"{ef_rank} qid:{d_id} {info}\n")

        df.to_excel(f"{sn}output.xlsx", index = True)
        print(f"在{datetime.datetime.now()}生成{sn}output.xlsx")
        # print(df)



# drug_id = readDrugName()
# print(drug_id)

df = pd.read_excel("drug_id.xlsx")

drug_em = {}
print(f"在{datetime.datetime.now()}开始生成药物向量")
for index, row in df.iterrows():
    drug_em[row[1]] = geteb(row[0])
    df.at[index, "Drug_Vector"] = str(drug_em[row[1]])
    # print(len(drug_em[row[1]]))

df.to_excel("drug_id_vector.xlsx", index = False)
print(f"药物向量在{datetime.datetime.now()}生成完毕")

addDrugNameAndVector(readDrugName(), drug_em)

