import pandas as pd

df = pd.read_excel('D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\ouput_features.xlsx')
# 获取列数
num_cols = len(df.columns)

# 合并每行的第1列到最后一列的数据，并添加列数和冒号
df['合并列'] = df.apply(lambda row: ' '.join([f'{i+1}:{value}' for i, value in enumerate(row)]), axis=1)

# 在合并后的数据前后加上引号
df['合并列'] = '"' + df['合并列'] + ' "'

# 删除原来的第1列到最后一列
df.drop(df.columns[0:num_cols], axis=1, inplace=True)

# 将处理后的数据保存为新的Excel文件
output_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\output_emb_合并.xlsx'
df.to_excel(output_path, index=False)

print("处理完成，结果已保存到:", output_path)





import pandas as pd

# 读取表格1数据
table1_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\output_emb_合并.xlsx'  # 表格1文件路径
table1_df = pd.read_excel(table1_path)

# 读取表格2数据
table2_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\dataset\\train_origin.xlsx'  # 表格2文件路径
table2_df = pd.read_excel(table2_path)

# 进行表格匹配
merged_df = pd.merge(table2_df, table1_df, left_on=table2_df.columns[3], right_on=table1_df.columns[0], how='left')

# 将匹配到的数据写入表格2的第四列
table2_df[table2_df.columns[4]] = merged_df[table1_df.columns[1]]

# 将结果保存到新的表格文件中
output_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\train_output.xlsx'
table2_df.to_excel(output_path, index=False)

print("匹配并写入完成，结果已保存到:", output_path)

# 读取表格1数据
table1_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\output_emb_合并.xlsx'  # 表格1文件路径
table1_df = pd.read_excel(table1_path)

# 读取表格2数据
table2_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\dataset\\test_origin.xlsx'  # 表格2文件路径
table2_df = pd.read_excel(table2_path)

# 进行表格匹配
merged_df = pd.merge(table2_df, table1_df, left_on=table2_df.columns[3], right_on=table1_df.columns[0], how='left')

# 将匹配到的数据写入表格2的第四列
table2_df[table2_df.columns[4]] = merged_df[table1_df.columns[1]]

# 将结果保存到新的表格文件中
output_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\test_output.xlsx'
table2_df.to_excel(output_path, index=False)

print("匹配并写入完成，结果已保存到:", output_path)




import pandas as pd
#
# 读取表格数据
table_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\test_output.xlsx'  # 表格文件路径
table_df = pd.read_excel(table_path)

output_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\test_dataset_magcl.txt'

# 打开 txt 文件准备写入数据
with open(output_path, 'w') as file:
    # 遍历表格的每一行数据
    for index, row in table_df.iterrows():
        # 将第一列的数据直接写入 txt 文件
        file.write(str(row[table_df.columns[0]]) + ' ')

        # 在第二列前面加上 'qid:'，然后写入 txt 文件
        file.write('qid:' + str(row[table_df.columns[1]]) + ' ')

        # 写入第三列的数据
        file.write(str(row[table_df.columns[2]]))

        # 写入换行符
        file.write('\n')

print("处理完成，结果已保存到:", output_path)


# 读取表格数据
table_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\train_output.xlsx'  # 表格文件路径
table_df = pd.read_excel(table_path)

output_path = 'D:\\yanjiu\\DrugRepSS\\embedding\\MAGCLemb\\result\\train_dataset_magcl.txt'

# 打开 txt 文件准备写入数据
with open(output_path, 'w') as file:
    # 遍历表格的每一行数据
    for index, row in table_df.iterrows():
        # 将第一列的数据直接写入 txt 文件
        file.write(str(row[table_df.columns[0]]) + ' ')

        # 在第二列前面加上 'qid:'，然后写入 txt 文件
        file.write('qid:' + str(row[table_df.columns[1]]) + ' ')

        # 写入第三列的数据
        file.write(str(row[table_df.columns[2]]))

        # 写入换行符
        file.write('\n')

print("处理完成，结果已保存到:", output_path)

