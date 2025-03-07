import os
import subprocess

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
        data_path='./ori_data/'+data_name+'.csv'
        df.to_csv(data_path, index=False)
        print(f"数据已保存为 {data_name}")

        print("启动 TimeGAN 训练...")
        s_data_name=data_name+'/'+data_name
        subprocess.run([
            "python", "main_timegan.py",
            "--data_name", s_data_name,
            "--seq_len", "24",
            "--module", "gru",
            "--hidden_dim", "24",
            "--num_layer", "3",
            "--iteration", "1000",
            "--batch_size", "128",
            "--train", "True",
            "--model_path", "./save_model/",
            "--target", target,
            "--feature", "S"
        ], check=True)
        print("TimeGAN 训练完成")

        # 返回成功响应
        return jsonify({"message": "Data received successfully", "row_count": len(df)}), 200

    except Exception as e:
        # 捕获所有异常并返回错误信息
        print(f"处理数据时出错: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 启动 Flask 服务器
    app.run(host='0.0.0.0', port=5001, debug=True)

