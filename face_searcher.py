"""
人脸搜索工具 - Face Searcher
功能：上传一张人脸照片，搜索指定文件夹中包含该人物的所有图片和视频
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import pickle
import json
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import sys

# 尝试导入 face_recognition，如果不可用则使用备用方案
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False
    print("警告: face_recognition 库未安装，将使用简单方案")

# ============ 配置 ============
APP_NAME = "人脸搜索工具"
APP_VERSION = "1.0"
CACHE_FILE = "face_cache.pkl"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.mp4', '.avi', '.mov', '.mkv']

# ============ 人脸识别核心 ============
class FaceSearcher:
    def __init__(self):
        self.known_encodings = []
        self.known_paths = []
        self.cache_loaded = False
    
    def extract_faces_from_folder(self, folder_path, progress_callback=None):
        """扫描文件夹，提取所有人脸特征"""
        self.known_encodings = []
        self.known_paths = []
        
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for f in filenames:
                if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                    files.append(os.path.join(root, f))
        
        total = len(files)
        for i, file_path in enumerate(files):
            try:
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # 视频文件：提取关键帧
                    encodings = self.extract_faces_from_video(file_path)
                else:
                    # 图片文件
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)
                
                for enc in encodings:
                    self.known_encodings.append(enc)
                    self.known_paths.append(file_path)
                
            except Exception as e:
                pass
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        self.save_cache(folder_path)
        return len(self.known_encodings)
    
    def extract_faces_from_video(self, video_path):
        """从视频中提取人脸"""
        encodings = []
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 每隔几帧取一次，避免太多
            step = max(1, frame_count // 10)
            
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_encs = face_recognition.face_encodings(rgb_frame)
                    encodings.extend(face_encs)
            cap.release()
        except:
            pass
        return encodings
    
    def search(self, target_image_path, threshold=0.6):
        """搜索目标人物"""
        if not self.known_encodings:
            return []
        
        target_image = face_recognition.load_image_file(target_image_path)
        target_encodings = face_recognition.face_encodings(target_image)
        
        if not target_encodings:
            return []
        
        target_encoding = target_encodings[0]
        results = []
        
        for i, known_enc in enumerate(self.known_encodings):
            distance = np.linalg.norm(known_enc - target_encoding)
            if distance < threshold:
                results.append({
                    'path': self.known_paths[i],
                    'distance': distance,
                    'count': results.count(lambda x: x['path'] == self.known_paths[i])
                })
        
        # 按路径分组，统计每个文件匹配到的人脸数
        path_counts = {}
        for r in results:
            path = r['path']
            if path not in path_counts:
                path_counts[path] = {'path': path, 'count': 0, 'min_distance': 1.0}
            path_counts[path]['count'] += 1
            path_counts[path]['min_distance'] = min(path_counts[path]['min_distance'], r['distance'])
        
        # 按匹配人脸数排序
        sorted_results = sorted(path_counts.values(), key=lambda x: (-x['count'], x['min_distance']))
        return sorted_results
    
    def save_cache(self, folder_path):
        """保存缓存"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_encodings,
                    'paths': self.known_paths,
                    'folder': folder_path
                }, f)
        except:
            pass
    
    def load_cache(self, folder_path):
        """加载缓存"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('folder') == folder_path:
                        self.known_encodings = data.get('encodings', [])
                        self.known_paths = data.get('paths', [])
                        return True
        except:
            pass
        return False

# ============ GUI 应用 ============
class FaceSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.searcher = FaceSearcher()
        self.target_image_path = None
        self.folder_path = ""
        self.search_results = []
        
        self.setup_ui()
        
        # 尝试设置主题
        try:
            style = ttk.Style()
            style.theme_use('clam')
        except:
            pass
    
    def setup_ui(self):
        """设置UI"""
        # 标题
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text=f"📸 {APP_NAME}", 
                               font=("微软雅黑", 20, "bold"),
                               bg="#2c3e50", fg="white")
        title_label.pack(pady=15)
        
        # 主内容区
        main_frame = tk.Frame(self.root, bg="#ecf0f1")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：控制面板
        control_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 目标图片
        tk.Label(control_frame, text="🎯 目标人物", font=("微软雅黑", 12, "bold"), 
                bg="white").pack(pady=(15, 5))
        
        self.target_image_label = tk.Label(control_frame, text="点击上传照片", 
                                          bg="#ecf0f1", width=25, height=10,
                                          relief=tk.SUNKEN, bd=1)
        self.target_image_label.pack(pady=5, padx=10)
        self.target_image_label.bind("<Button-1>", self.upload_target_image)
        
        upload_btn = tk.Button(control_frame, text="📁 上传目标照片", 
                              command=self.upload_target_image,
                              bg="#3498db", fg="white", relief=tk.RAISED,
                              font=("微软雅黑", 10))
        upload_btn.pack(pady=5, padx=10, fill=tk.X)
        
        # 文件夹选择
        tk.Label(control_frame, text="📂 搜索目录", font=("微软雅黑", 12, "bold"), 
                bg="white").pack(pady=(20, 5))
        
        self.folder_label = tk.Label(control_frame, text="未选择文件夹", 
                                      bg="#ecf0f1", width=25, fg="#7f8c8d",
                                      relief=tk.SUNKEN, bd=1)
        self.folder_label.pack(pady=5, padx=10)
        
        folder_btn = tk.Button(control_frame, text="📂 选择文件夹", 
                              command=self.select_folder,
                              bg="#27ae60", fg="white", relief=tk.RAISED,
                              font=("微软雅黑", 10))
        folder_btn.pack(pady=5, padx=10, fill=tk.X)
        
        # 扫描按钮
        self.scan_btn = tk.Button(control_frame, text="🔍 扫描人脸库", 
                                 command=self.start_scan,
                                 bg="#e67e22", fg="white", relief=tk.RAISED,
                                 font=("微软雅黑", 11, "bold"))
        self.scan_btn.pack(pady=15, padx=10, fill=tk.X)
        
        # 搜索按钮
        self.search_btn = tk.Button(control_frame, text="🚀 开始搜索", 
                                   command=self.start_search,
                                   bg="#e74c3c", fg="white", relief=tk.RAISED,
                                   font=("微软雅黑", 11, "bold"),
                                   state=tk.DISABLED)
        self.search_btn.pack(pady=(0, 15), padx=10, fill=tk.X)
        
        # 状态显示
        self.status_label = tk.Label(control_frame, text="就绪", 
                                    bg="white", fg="#7f8c8d", font=("微软雅黑", 9))
        self.status_label.pack(pady=5)
        
        # 右侧：结果展示
        result_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(result_frame, text="📋 搜索结果", font=("微软雅黑", 12, "bold"), 
                bg="white").pack(pady=10)
        
        # 结果列表（带滚动条）
        scroll_y = tk.Scrollbar(result_frame)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_listbox = tk.Listbox(result_frame, yscrollcommand=scroll_y.set,
                                         font=("微软雅黑", 10), selectbackground="#3498db")
        self.result_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        scroll_y.config(command=self.result_listbox.yview)
        
        self.result_listbox.bind("<Double-Button-1>", self.open_file)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='determinate', length=200)
        self.progress.pack(fill=tk.X, pady=(5, 0))
        self.progress.pack_forget()
    
    def upload_target_image(self, event=None):
        """上传目标图片"""
        path = filedialog.askopenfilename(
            title="选择目标人物照片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if path:
            self.target_image_path = path
            self.show_image(path, self.target_image_label)
            self.status_label.config(text=f"已选择: {os.path.basename(path)}")
            self.check_search_ready()
    
    def show_image(self, path, label, max_size=None):
        """显示图片"""
        try:
            img = Image.open(path)
            img.thumbnail((200, 150) if not max_size else max_size)
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label.image = photo
        except Exception as e:
            label.config(text=f"无法加载图片")
    
    def select_folder(self):
        """选择文件夹"""
        path = filedialog.askdirectory(title="选择照片文件夹")
        if path:
            self.folder_path = path
            self.folder_label.config(text=os.path.basename(path)[:20] + "..." if len(os.path.basename(path)) > 20 else os.path.basename(path))
            self.status_label.config(text=f"已选择: {path}")
            
            # 检查是否有缓存
            if self.searcher.load_cache(path):
                self.status_label.config(text="已加载缓存")
                self.check_search_ready()
            else:
                self.scan_btn.config(state=tk.NORMAL)
                self.status_label.config(text="需要扫描人脸库")
    
    def check_search_ready(self):
        """检查是否准备好搜索"""
        if self.target_image_path and self.folder_path and self.searcher.known_encodings:
            self.search_btn.config(state=tk.NORMAL)
        else:
            self.search_btn.config(state=tk.DISABLED)
    
    def start_scan(self):
        """开始扫描"""
        if not self.folder_path:
            messagebox.showwarning("提示", "请先选择文件夹")
            return
        
        self.scan_btn.config(state=tk.DISABLED)
        self.search_btn.config(state=tk.DISABLED)
        self.progress.pack(fill=tk.X, pady=(5, 0))
        
        def scan_thread():
            def progress_callback(current, total):
                self.root.after(0, lambda: self.update_progress(current, total))
            
            count = self.searcher.extract_faces_from_folder(self.folder_path, progress_callback)
            self.root.after(0, lambda: self.scan_complete(count))
        
        thread = threading.Thread(target=scan_thread, daemon=True)
        thread.start()
    
    def update_progress(self, current, total):
        """更新进度条"""
        self.progress['value'] = (current / total) * 100
        self.status_label.config(text=f"扫描中: {current}/{total}")
    
    def scan_complete(self, count):
        """扫描完成"""
        self.progress.pack_forget()
        self.progress['value'] = 0
        self.scan_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"扫描完成，找到 {count} 个人脸")
        messagebox.showinfo("完成", f"扫描完成！\n共提取 {count} 个人脸特征")
        self.check_search_ready()
    
    def start_search(self):
        """开始搜索"""
        if not self.target_image_path:
            messagebox.showwarning("提示", "请先上传目标人物照片")
            return
        
        self.search_btn.config(state=tk.DISABLED)
        self.result_listbox.delete(0, tk.END)
        self.result_listbox.insert(tk.END, "正在搜索...")
        
        def search_thread():
            results = self.searcher.search(self.target_image_path)
            self.root.after(0, lambda: self.search_complete(results))
        
        thread = threading.Thread(target=search_thread, daemon=True)
        thread.start()
    
    def search_complete(self, results):
        """搜索完成"""
        self.search_btn.config(state=tk.NORMAL)
        self.result_listbox.delete(0, tk.END)
        
        if not results:
            self.result_listbox.insert(tk.END, "未找到匹配结果")
            return
        
        self.search_results = results
        self.result_listbox.insert(tk.END, f"找到 {len(results)} 个匹配文件:\n")
        
        for r in results[:100]:  # 最多显示100个
            name = os.path.basename(r['path'])
            count = r['count']
            self.result_listbox.insert(tk.END, f"  📷 {name} (匹配{count}次)")
        
        if len(results) > 100:
            self.result_listbox.insert(tk.END, f"\n... 还有 {len(results) - 100} 个结果")
    
    def open_file(self, event=None):
        """打开文件"""
        selection = self.result_listbox.curselection()
        if selection:
            content = self.result_listbox.get(selection[0])
            if content.startswith("  📷"):
                # 提取文件路径
                for r in self.search_results:
                    if os.path.basename(r['path']) in content:
                        try:
                            os.startfile(r['path']) if os.name == 'nt' else subprocess.run(['open', r['path']])
                        except:
                            messagebox.showerror("错误", f"无法打开: {r['path']}")
                        break

# ============ 主程序 ============
def main():
    # 检查依赖
    if not USE_FACE_RECOGNITION:
        messagebox.showerror("依赖缺失", 
            "请先安装必要的库:\n\npip install face_recognition dlib numpy opencv-python Pillow\n\n或者下载预编译的 dlib wheel 文件")
        return
    
    root = tk.Tk()
    app = FaceSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
