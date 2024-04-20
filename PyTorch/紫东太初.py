# -*- coding: utf-8 -*-
import requests

if __name__ == '__main__':
    params = {
        'api_key': 'x892rya230yb2dfgaexfdw7m',
        'model_code': 'taichu_llm',
        'question': '编写一个程序，输入一个字符串，判断该字符串是否是回文字符串',
        # 'stream_format': 'json'
        'do_stream': False
    }
    api = 'https://ai-maas.wair.ac.cn/maas/v1/model_api/invoke'
    response = requests.post(api, json=params)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        # for line in response.iter_lines(decode_unicode=True):
        #    print(line)
        print(response.json()['data']['content'])
    else:
        print('failed')