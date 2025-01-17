import os
import time
import json
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Flask 应用配置
app = Flask(__name__)
socketio = SocketIO(app)

# 监控的图片目录
DEBUG_DIR = "./debug"
LOSS_FILE = os.path.join(DEBUG_DIR, "loss_data.txt")
POINT_NUM = os.path.join(DEBUG_DIR, "point_num.txt")
os.makedirs(DEBUG_DIR, exist_ok=True)

@app.route("/")
def index():
    """前端主页面"""
    return render_template("index.html")

@app.route("/debug/<filename>")
def get_image(filename):
    """提供图片访问接口"""
    return send_from_directory(DEBUG_DIR, filename)

@socketio.on("connect")
def handle_connect():
    print("Client connected!")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected!")

def push_image(image_path):
    """推送图片更新到前端"""
    socketio.emit("update_image", {
        "image_path": f"/debug/{os.path.basename(image_path)}",
        "image_name": os.path.basename(image_path)
    })

def push_loss_data():
    try:
        with open(LOSS_FILE, "r") as f:
            lines = f.readlines()
        if not lines:
            return
        
        # 第一行是标签
        labels = lines[0].strip().split(",")
        
        # 后面的行是数据
        data = [list(map(float, line.strip().split(","))) for line in lines[1:]]
        
        # 将标签和数据一起发送
        socketio.emit("update_loss", {"labels": labels, "data": data})
    except Exception as e:
        print(f"Error reading loss file: {e}")

def push_point_num_data():
    try:
        with open(POINT_NUM, "r") as f:
            lines = f.readlines()
        if not lines:
            return
        
        # 第一行是标签
        labels = lines[0].strip().split(",")
        
        # 后面的行是数据
        data = [list(map(float, line.strip().split(","))) for line in lines[1:]]
        
        # 将标签和数据一起发送
        socketio.emit("update_point_num", {"labels": labels, "data": data})
    except Exception as e:
        print(f"Error reading point_num file: {e}")

# 文件夹监控
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            print(f"New image detected: {event.src_path}")
            push_image(event.src_path)
        elif event.src_path == LOSS_FILE:
            push_loss_data()
        elif event.src_path == POINT_NUM:
            push_point_num_data()

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            print(f"Image updated: {event.src_path}")
            push_image(event.src_path)
        elif event.src_path == LOSS_FILE:
            push_loss_data()
        elif event.src_path == POINT_NUM:
            push_point_num_data()

def start_folder_monitoring():
    """启动文件夹监控"""
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, DEBUG_DIR, recursive=False)
    observer.start()
    print(f"Monitoring folder: {DEBUG_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # 启动文件夹监控进程
    from threading import Thread
    monitor_thread = Thread(target=start_folder_monitoring, daemon=True)
    monitor_thread.start()

    # 启动 Flask 服务器
    socketio.run(app, host="0.0.0.0", port=5000)
