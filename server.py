import os

from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_data():
    """
    接收客户端发送的数据，并返回处理结果。
    """
    try:
        # 接收 JSON 数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # 将数据转换为 Pandas DataFrame（如果需要进一步处理）
        data_name=data.get('data_name')
        target=data.get('target')
        result_data=data.get('result_data')
        if not data_name or not target or not result_data:
            return jsonify({"error": "Incomplete data"}), 400

        df = pd.DataFrame(result_data)
        print("接收到的数据:")
        print(f"Received data name: {data_name}")
        print(f"Target feature: {target}")
        print(f"Received data records: {len(df)}")

        # 可以在这里对数据进行处理，例如保存到文件或数据库
        # 保存为 CSV 文件
        data_path=os.path.join('./data',data_name)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        f_data_name=data_path+'/'+data_name+'.csv'
        df.to_csv(f_data_name, index=False)
        print(f"数据已保存为 {f_data_name}")

        # 返回成功响应
        return jsonify({"message": "Data received successfully", "row_count": len(df)}), 200

    except Exception as e:
        # 捕获所有异常并返回错误信息
        print(f"处理数据时出错: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=5001, debug=True)

