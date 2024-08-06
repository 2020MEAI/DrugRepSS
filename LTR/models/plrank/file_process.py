import pandas as pd

# 读取表格数据
table_path = 'output测试集pl.xlsx'  # 表格文件路径
table_df = pd.read_excel(table_path)

output_path = 'output测试集pl.txt'

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
