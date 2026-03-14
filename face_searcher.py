"""
人脸搜索工具 - Face Searcher (轻量版)
功能：上传一张人脸照片，搜索指定文件夹中包含该人物的所有图片
依赖：opencv-python, numpy, Pillow (都是标准库，容易打包)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import pickle
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import hashlib

# ============ 配置 ============
APP_NAME = "人脸搜索工具"
APP_VERSION = "1.1"
CACHE_FILE = "face_cache.pkl"

# OpenCV 内置的 Haar Cascade 人脸检测器（无需额外下载）
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# ============ 人脸识别核心 ============
class FaceSearcher:
    def __init__(self):
        self.known_faces = []  # 存储 (人脸特征, 文件路径)
        self.folder_path = ""
    
    def get_face_feature(self, image):
        """提取人脸特征（使用简单的图像哈希）"""
        if image is None:
            return None
        
        # 转为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 人脸检测
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # 提取人脸区域
        face_roi = gray[y:y+h, x:x+w]
        
        # 调整大小并计算特征
        face_resized = cv2.resize(face_roi, (100, 100))
        
        # 使用 HOG 特征或其他简单特征
        features = face_resized.flatten()
        
        return features
    
    def extract_faces_from_folder(self, folder_path, progress_callback=None):
        """扫描文件夹，提取所有人脸"""
        self.known_faces = []
        self.folder_path = folder_path
        
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for f in filenames:
                if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                    files.append(os.path.join(root, f))
        
        total = len(files)
        
        for i, file_path in enumerate(files):
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    feature = self.get_face_feature(img)
                    if feature is not None:
                        self.known_faces.append((feature, file_path))
            except:
                pass
            
            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total)
        
        self.save_cache()
        return len(self.known_faces)
    
    def search(self, target_image_path, threshold=0.7):
        """搜索目标人物"""
        if not self.known_faces:
            return []
        
        target_img = cv2.imread(target_image_path)
        if target_img is None:
            return []
        
        target_feature = self.get_face_feature(target_img)
        if target_feature is None:
            messagebox.showwarning("未检测到人脸", "目标图片中未检测到人脸，请换一张照片")
            return []
        
        results = []
        
        for known_feature, file_path in self.known_faces:
            # 计算相似度
            corr = np.corrcoef(target_feature, known_feature)[0, 1]
            
            if corr > threshold:
                results.append({
                    'path': file_path,
                    'similarity': corr
                })
        
        # 按相似度排序
        results.sort(key=lambda x: -x['similarity'])
        
        # 按文件分组
        path_groups = {}
        for r in results:
            path = r['path']
            if path not in path_groups:
                path_groups[path] = {'path': path, 'count': 0, 'max_similarity': 0}
            path_groups[path]['count'] += 1
            path_groups[path]['max_similarity'] = max(path_groups[path]['max_similarity'], r['similarity'])
        
        sorted_results = sorted(path_groups.values(), key=lambda x: -x['max_similarity'])
        return sorted_results
    
    def save_cache(self):
        """保存缓存"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump({
                    'faces': self.known_faces,
                    'folder': self.folder_path
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
                        self.known_faces = data.get('faces', [])
                        self.folder_path = folder_path
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
        
        # 结果列表
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
            img.thumbnail((200, 150))
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label.image = photo
        except Exception as e:
            label.config(text=f"无法加载")
    
    def select_folder(self):
        """选择文件夹"""
        path = filedialog.askdirectory(title="选择照片文件夹")
        if path:
            self.folder_path = path
            name = os.path.basename(path)
            self.folder_label.config(text=name[:20] + "..." if len(name) > 20 else name)
            self.status_label.config(text=f"已选择: {path}")
            
            if self.searcher.load_cache(path):
                self.status_label.config(text="已加载缓存")
                self.check_search_ready()
            else:
                self.scan_btn.config(state=tk.NORMAL)
                self.status_label.config(text="需要扫描人脸库")
    
    def check_search_ready(self):
        """检查是否准备好搜索"""
        if self.target_image_path and self.folder_path and self.searcher.known_faces:
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
        
        for r in results[:100]:
            name = os.path.basename(r['path'])
            sim = int(r['max_similarity'] * 100)
            self.result_listbox.insert(tk.END, f"  📷 {name} (相似度{sim}%)")
        
        if len(results) > 100:
            self.result_listbox.insert(tk.END, f"\n... 还有 {len(results) - 100} 个结果")
    
    def open_file(self, event=None):
        """打开文件"""
        selection = self.result_listbox.curselection()
        if selection:
            content = self.result_listbox.get(selection[0])
            if content.startswith("  📷"):
                for r in self.search_results:
                    if os.path.basename(r['path']) in content:
                        try:
                            os.startfile(r['path']) if os.name == 'nt' else subprocess.run(['open', r['path']])
                        except:
                            messagebox.showerror("错误", f"无法打开: {r['path']}")
                        break

# ============ 主程序 ============
def main():
    root = tk.Tk()
    app = FaceSearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
