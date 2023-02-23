import time

import os

filename = '/home/asd/PycharmProjects/pythonProject1/Learning_Control_System/logs/log_cp_error_0_epoch_100_iter_100_ver_2.txt'

last_updated = 0

while True:

    # 获取文件最后修改时间

    updated = os.path.getmtime(filename)

    # 如果文件已经更新，打印新内容

    if updated > last_updated:
        last_updated = updated

        with open(filename, 'r') as file:
            content = file.read()

            print(content)

    time.sleep(60)  # 每秒钟检查一次
