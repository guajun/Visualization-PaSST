"""
PaSST 注意力可视化视频生成器
在频谱图上叠加注意力热度图，生成类似 DINO 风格的可视化视频
"""

import os
import warnings
# 抑制 hear21passt 库的警告
warnings.filterwarnings("ignore", message=".*autocast.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Input image size.*doesn't match model.*")

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import librosa
import cv2
import subprocess
import tempfile
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from hear21passt.base import get_basic_model


@dataclass
class Config:
    """配置类"""
    # 输入输出
    audio_path: str
    video_path: str
    save_frames: bool
    frames_dir: str
    
    # 范围
    range_enabled: bool
    start_time: float
    end_time: float
    
    # 视频
    video_width: int
    video_height: int
    fps: int
    codec: str
    
    # 音频
    sample_rate: int
    window_duration: float
    hop_duration: float
    
    # 频谱图
    n_mels: int
    n_fft: int
    hop_length: int
    fmin: int
    fmax: int
    spec_colormap: str
    
    # 注意力
    layer_index: int
    head_reduction: str
    specific_head: int
    attn_colormap: str
    
    # 叠加
    overlay_mode: str
    attention_weight: float
    spectrogram_weight: float
    final_colormap: str
    
    # 布局
    layout_mode: str
    show_spectrogram: bool
    show_attention: bool
    show_overlay: bool
    margin_top: int
    margin_bottom: int
    margin_left: int
    margin_right: int
    
    # 标注
    show_title: bool
    title: str
    show_timestamp: bool
    show_colorbar: bool
    show_labels: bool
    font_size: int
    font_color: str
    
    # 模型
    device: str
    
    # 性能
    use_amp: bool
    cache_attention: bool
    
    # 音频叠加
    add_audio: bool
    
    # 播放头（时间指示线）
    show_playhead: bool
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            # 输入输出
            audio_path=cfg['input']['audio_path'],
            video_path=cfg['output']['video_path'],
            save_frames=cfg['output']['save_frames'],
            frames_dir=cfg['output']['frames_dir'],
            
            # 范围
            range_enabled=cfg['range']['enabled'],
            start_time=cfg['range']['start_time'],
            end_time=cfg['range']['end_time'],
            
            # 视频
            video_width=cfg['video']['width'],
            video_height=cfg['video']['height'],
            fps=cfg['video']['fps'],
            codec=cfg['video']['codec'],
            
            # 音频
            sample_rate=cfg['audio']['sample_rate'],
            window_duration=cfg['audio']['window_duration'],
            hop_duration=cfg['audio']['hop_duration'],
            
            # 频谱图
            n_mels=cfg['spectrogram']['n_mels'],
            n_fft=cfg['spectrogram']['n_fft'],
            hop_length=cfg['spectrogram']['hop_length'],
            fmin=cfg['spectrogram']['fmin'],
            fmax=cfg['spectrogram']['fmax'],
            spec_colormap=cfg['spectrogram']['colormap'],
            
            # 注意力
            layer_index=cfg['attention']['layer_index'],
            head_reduction=cfg['attention']['head_reduction'],
            specific_head=cfg['attention']['specific_head'],
            attn_colormap=cfg['attention']['colormap'],
            
            # 叠加
            overlay_mode=cfg['overlay']['mode'],
            attention_weight=cfg['overlay']['attention_weight'],
            spectrogram_weight=cfg['overlay']['spectrogram_weight'],
            final_colormap=cfg['overlay']['final_colormap'],
            
            # 布局
            layout_mode=cfg['layout']['mode'],
            show_spectrogram=cfg['layout']['split']['show_spectrogram'],
            show_attention=cfg['layout']['split']['show_attention'],
            show_overlay=cfg['layout']['split']['show_overlay'],
            margin_top=cfg['layout']['margin']['top'],
            margin_bottom=cfg['layout']['margin']['bottom'],
            margin_left=cfg['layout']['margin']['left'],
            margin_right=cfg['layout']['margin']['right'],
            
            # 标注
            show_title=cfg['annotation']['show_title'],
            title=cfg['annotation']['title'],
            show_timestamp=cfg['annotation']['show_timestamp'],
            show_colorbar=cfg['annotation']['show_colorbar'],
            show_labels=cfg['annotation']['show_labels'],
            font_size=cfg['annotation']['font_size'],
            font_color=cfg['annotation']['font_color'],
            
            # 模型
            device=cfg['model']['device'],
            
            # 性能
            use_amp=cfg['performance']['use_amp'],
            cache_attention=cfg['performance']['cache_attention'],
            
            # 音频叠加
            add_audio=cfg['output'].get('add_audio', True),
            
            # 播放头
            show_playhead=cfg['annotation'].get('show_playhead', True),
        )


class AttentionExtractor:
    """提取 PaSST 注意力权重"""
    
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.attention_maps: List[torch.Tensor] = []
        self.hooks: List[Any] = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """设置前向钩子以捕获注意力权重"""
        def create_hook():
            def hook(module, input, output):
                # 获取输入
                x = input[0]
                B, N, C = x.shape
                
                # 计算注意力权重
                qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                
                self.attention_maps.append(attn.detach().cpu())
            return hook
        
        # 为每个 Block 的 Attention 注册钩子
        for block in self.model.net.blocks:
            hook = block.attn.register_forward_hook(create_hook())
            self.hooks.append(hook)
    
    def clear(self):
        """清空注意力缓存"""
        self.attention_maps.clear()
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_attention_map(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """
        获取处理后的注意力图
        
        Args:
            grid_size: (freq_patches, time_patches)
        
        Returns:
            注意力热度图 numpy 数组
        """
        if not self.attention_maps:
            return None
        
        # 选择层
        layer_idx = self.config.layer_index
        if layer_idx < 0:
            layer_idx = len(self.attention_maps) + layer_idx
        
        attn = self.attention_maps[layer_idx]  # [B, num_heads, N, N]
        
        # 处理多头
        if self.config.head_reduction == "mean":
            attn = attn.mean(dim=1)  # [B, N, N]
        elif self.config.head_reduction == "max":
            attn = attn.max(dim=1)[0]  # [B, N, N]
        else:  # specific
            attn = attn[:, self.config.specific_head]  # [B, N, N]
        
        # 提取 CLS token 对 patch tokens 的注意力
        # PaSST 有 2 个特殊 token: CLS 和 DIST
        num_special_tokens = 2
        cls_attn = attn[0, 0, num_special_tokens:]  # [num_patches]
        
        # 重塑为 2D 网格
        freq_patches, time_patches = grid_size
        num_patches = freq_patches * time_patches
        
        if len(cls_attn) >= num_patches:
            attn_map = cls_attn[:num_patches].reshape(freq_patches, time_patches).numpy()
        else:
            # 如果 patch 数量不匹配，尝试推断
            attn_map = cls_attn.reshape(-1, time_patches).numpy()
        
        return attn_map


class SpectrogramGenerator:
    """生成梅尔频谱图"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate(self, audio: np.ndarray) -> np.ndarray:
        """生成梅尔频谱图"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db


class FrameRenderer:
    """渲染视频帧"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dpi = 100
        self.fig_width = config.video_width / self.dpi
        self.fig_height = config.video_height / self.dpi
    
    def render_single_mode(
        self,
        spectrogram: np.ndarray,
        attention_map: np.ndarray,
        current_time: float,
        total_duration: float,
        window_duration: float = 10.0
    ) -> np.ndarray:
        """渲染单一叠加模式的帧"""
        fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)
        fig.patch.set_facecolor('black')
        
        # 调整注意力图大小以匹配频谱图
        attn_resized = cv2.resize(
            attention_map, 
            (spectrogram.shape[1], spectrogram.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # 归一化
        spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # 叠加
        if self.config.overlay_mode == "blend":
            overlay = (spec_norm * self.config.spectrogram_weight + 
                      attn_norm * self.config.attention_weight)
        elif self.config.overlay_mode == "multiply":
            overlay = spec_norm * attn_norm
        elif self.config.overlay_mode == "screen":
            overlay = 1 - (1 - spec_norm) * (1 - attn_norm)
        else:
            overlay = spec_norm * self.config.spectrogram_weight + attn_norm * self.config.attention_weight
        
        # 创建绘图区域
        ax = fig.add_axes([
            self.config.margin_left / self.config.video_width,
            self.config.margin_bottom / self.config.video_height,
            1 - (self.config.margin_left + self.config.margin_right) / self.config.video_width,
            1 - (self.config.margin_top + self.config.margin_bottom) / self.config.video_height
        ])
        
        # 绘制叠加图
        im = ax.imshow(
            overlay, 
            aspect='auto', 
            origin='lower', 
            cmap=self.config.final_colormap,
            vmin=0, vmax=1
        )
        
        # 添加播放头指示线（居中的垂直线）
        if self.config.show_playhead:
            center_x = spectrogram.shape[1] / 2
            ax.axvline(x=center_x, color='white', linewidth=2, alpha=0.8)
            # 添加三角形标记
            ax.plot(center_x, spectrogram.shape[0] - 1, 'v', color='white', markersize=10, alpha=0.8)
            ax.plot(center_x, 0, '^', color='white', markersize=10, alpha=0.8)
        
        # 计算时间轴标签（以当前时间为中心）
        n_ticks = 5
        time_width = overlay.shape[1]
        tick_positions = np.linspace(0, time_width - 1, n_ticks)
        # 当前时间在中心，计算窗口的时间范围
        window_start_time = current_time - window_duration / 2
        window_end_time = current_time + window_duration / 2
        tick_times = np.linspace(window_start_time, window_end_time, n_ticks)
        tick_labels = [f"{t:.1f}s" for t in tick_times]
        
        # 设置标签
        if self.config.show_labels:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel('Time', color=self.config.font_color, fontsize=self.config.font_size)
            ax.set_ylabel('Frequency', color=self.config.font_color, fontsize=self.config.font_size)
            ax.tick_params(colors=self.config.font_color, labelsize=self.config.font_size - 4)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 标题
        if self.config.show_title:
            ax.set_title(
                self.config.title, 
                color=self.config.font_color, 
                fontsize=self.config.font_size + 4,
                pad=10
            )
        
        # 时间戳
        if self.config.show_timestamp:
            time_str = f"Time: {current_time:.2f}s / {total_duration:.2f}s"
            fig.text(
                0.5, 0.02, time_str,
                ha='center', va='bottom',
                color=self.config.font_color,
                fontsize=self.config.font_size,
                fontweight='bold'
            )
        
        # 颜色条
        if self.config.show_colorbar:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.ax.yaxis.set_tick_params(color=self.config.font_color)
            cbar.outline.set_edgecolor(self.config.font_color)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.font_color)
        
        # 转换为 numpy 数组
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = frame[:, :, :3]  # 去掉 alpha 通道，保留 RGB
        
        plt.close(fig)
        
        return frame
    
    def render_split_mode(
        self,
        spectrogram: np.ndarray,
        attention_map: np.ndarray,
        current_time: float,
        total_duration: float
    ) -> np.ndarray:
        """渲染分屏模式的帧"""
        fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)
        fig.patch.set_facecolor('black')
        
        # 计算要显示的面板数
        panels = []
        if self.config.show_spectrogram:
            panels.append(('spectrogram', spectrogram, self.config.spec_colormap))
        if self.config.show_attention:
            attn_resized = cv2.resize(attention_map, (spectrogram.shape[1], spectrogram.shape[0]))
            panels.append(('attention', attn_resized, self.config.attn_colormap))
        if self.config.show_overlay:
            attn_resized = cv2.resize(attention_map, (spectrogram.shape[1], spectrogram.shape[0]))
            spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
            attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            overlay = spec_norm * self.config.spectrogram_weight + attn_norm * self.config.attention_weight
            panels.append(('overlay', overlay, self.config.final_colormap))
        
        n_panels = len(panels)
        if n_panels == 0:
            n_panels = 1
            panels = [('overlay', spectrogram, self.config.final_colormap)]
        
        # 创建子图
        for i, (name, data, cmap) in enumerate(panels):
            ax = fig.add_subplot(1, n_panels, i + 1)
            
            # 归一化数据
            if name != 'spectrogram':
                data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap)
            
            titles = {'spectrogram': 'Spectrogram', 'attention': 'Attention', 'overlay': 'Overlay'}
            ax.set_title(titles.get(name, name), color=self.config.font_color, fontsize=self.config.font_size)
            
            if self.config.show_labels:
                ax.set_xlabel('Time', color=self.config.font_color, fontsize=self.config.font_size - 2)
                if i == 0:
                    ax.set_ylabel('Frequency', color=self.config.font_color, fontsize=self.config.font_size - 2)
                ax.tick_params(colors=self.config.font_color, labelsize=self.config.font_size - 6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 主标题
        if self.config.show_title:
            fig.suptitle(
                self.config.title,
                color=self.config.font_color,
                fontsize=self.config.font_size + 4,
                y=0.98
            )
        
        # 时间戳
        if self.config.show_timestamp:
            time_str = f"Time: {current_time:.2f}s / {total_duration:.2f}s"
            fig.text(
                0.5, 0.02, time_str,
                ha='center', va='bottom',
                color=self.config.font_color,
                fontsize=self.config.font_size,
                fontweight='bold'
            )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 转换为 numpy 数组
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frame = frame[:, :, :3]  # 去掉 alpha 通道，保留 RGB
        
        plt.close(fig)
        
        return frame
    
    def render(
        self,
        spectrogram: np.ndarray,
        attention_map: np.ndarray,
        current_time: float,
        total_duration: float,
        window_duration: float = 10.0
    ) -> np.ndarray:
        """根据配置渲染帧"""
        if self.config.layout_mode == "single":
            return self.render_single_mode(spectrogram, attention_map, current_time, total_duration, window_duration)
        else:
            return self.render_split_mode(spectrogram, attention_map, current_time, total_duration)


class VideoGenerator:
    """视频生成器主类"""
    
    def __init__(self, config_path: str):
        self.config = Config.from_yaml(config_path)
        self.model = None
        self.attention_extractor = None
        self.spectrogram_generator = SpectrogramGenerator(self.config)
        self.frame_renderer = FrameRenderer(self.config)
    
    def _load_model(self):
        """加载 PaSST 模型"""
        print("Loading PaSST model...")
        self.model = get_basic_model(mode="logits")
        self.model.eval()
        
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.attention_extractor = AttentionExtractor(self.model, self.config)
        print("Model loaded successfully!")
    
    def _load_audio(self) -> Tuple[np.ndarray, float]:
        """加载音频文件"""
        print(f"Loading audio: {self.config.audio_path}")
        audio, _ = librosa.load(self.config.audio_path, sr=self.config.sample_rate)
        
        # 应用范围选择
        if self.config.range_enabled:
            start_sample = int(self.config.start_time * self.config.sample_rate)
            
            if self.config.end_time < 0:
                end_sample = len(audio)
            else:
                end_sample = int(self.config.end_time * self.config.sample_rate)
            
            audio = audio[start_sample:end_sample]
            print(f"Selected range: {self.config.start_time}s - {end_sample / self.config.sample_rate:.2f}s")
        
        duration = len(audio) / self.config.sample_rate
        print(f"Audio duration: {duration:.2f}s")
        
        return audio, duration
    
    def _get_grid_size(self) -> Tuple[int, int]:
        """获取 patch 网格大小"""
        # PaSST 默认配置
        # 对于 10 秒音频，频率方向 8 个 patch，时间方向 100 个 patch
        # 这取决于具体的模型配置
        try:
            grid_size = self.model.net.patch_embed.grid_size
            return grid_size
        except:
            # 默认值
            return (8, 100)
    
    def generate(self):
        """生成可视化视频"""
        # 加载模型和音频
        self._load_model()
        audio, total_duration = self._load_audio()
        
        # 计算帧数和时间步
        window_samples = int(self.config.window_duration * self.config.sample_rate)
        half_window_samples = window_samples // 2
        hop_samples = int(self.config.hop_duration * self.config.sample_rate)
        
        # 在音频前后添加填充，使得当前时间可以居中显示
        audio_padded = np.pad(audio, (half_window_samples, half_window_samples), mode='constant')
        
        # 计算总帧数 (基于原始音频长度)
        num_frames = len(audio) // hop_samples
        
        # 创建视频写入器
        if self.config.save_frames:
            os.makedirs(self.config.frames_dir, exist_ok=True)
        
        # 使用临时文件保存无音频视频
        temp_video_path = self.config.video_path + ".temp.mp4" if self.config.add_audio else self.config.video_path
        
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        out = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            self.config.fps,
            (self.config.video_width, self.config.video_height)
        )
        
        print(f"Generating {num_frames} frames...")
        grid_size = self._get_grid_size()
        
        # 生成每一帧
        for frame_idx in tqdm(range(num_frames), desc="Rendering"):
            # 当前时间点（相对于原始音频）
            current_sample = frame_idx * hop_samples
            
            # 在填充后的音频中，窗口以当前时间为中心
            # 窗口起始 = 当前采样点（在原始音频中）+ 0（因为我们加了 half_window 填充）
            # 这样窗口中心就是 current_sample + half_window = 原始音频的 current_sample
            start_sample = current_sample  # 在 padded audio 中的起始位置
            end_sample = start_sample + window_samples
            
            # 提取音频片段
            audio_chunk = audio_padded[start_sample:end_sample]
            
            # 计算当前时间（用于显示）
            current_time = current_sample / self.config.sample_rate
            if self.config.range_enabled:
                current_time += self.config.start_time
            
            # 生成频谱图
            spectrogram = self.spectrogram_generator.generate(audio_chunk)
            
            # 清空之前的注意力图
            self.attention_extractor.clear()
            
            # 模型推理
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
                if self.config.device == "cuda" and torch.cuda.is_available():
                    audio_tensor = audio_tensor.cuda()
                
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        _ = self.model(audio_tensor)
                else:
                    _ = self.model(audio_tensor)
            
            # 获取注意力图
            attention_map = self.attention_extractor.get_attention_map(grid_size)
            
            if attention_map is None:
                print(f"Warning: No attention map for frame {frame_idx}")
                continue
            
            # 渲染帧
            frame = self.frame_renderer.render(
                spectrogram,
                attention_map,
                current_time,
                total_duration + (self.config.start_time if self.config.range_enabled else 0),
                self.config.window_duration
            )
            
            # 转换 RGB 到 BGR (OpenCV 格式)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 写入视频
            out.write(frame_bgr)
            
            # 保存单独帧
            if self.config.save_frames:
                frame_path = os.path.join(self.config.frames_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_path, frame_bgr)
        
        # 释放资源
        out.release()
        self.attention_extractor.remove_hooks()
        
        # 添加背景音乐
        if self.config.add_audio:
            self._add_audio_to_video(temp_video_path, self.config.video_path, total_duration)
            # 删除临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        print(f"Video saved to: {self.config.video_path}")
    
    def _add_audio_to_video(self, video_path: str, output_path: str, duration: float):
        """使用 ffmpeg 将音频添加到视频"""
        print("Adding audio to video...")
        
        # 构建 ffmpeg 命令
        audio_start = self.config.start_time if self.config.range_enabled else 0
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(audio_start),
            "-t", str(duration),
            "-i", self.config.audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path
        ]
        
        try:
            # 使用 bytes 模式避免 Windows 编码问题
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True
            )
            print("Audio added successfully!")
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
            print(f"Warning: Failed to add audio. ffmpeg error: {stderr_msg}")
            print("Saving video without audio...")
            # 如果 ffmpeg 失败，直接重命名临时文件
            import shutil
            shutil.move(video_path, output_path)
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Saving video without audio...")
            print("Please install ffmpeg to enable audio support.")
            import shutil
            shutil.move(video_path, output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PaSST Attention Visualization Video Generator")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # 生成视频
    generator = VideoGenerator(args.config)
    generator.generate()


if __name__ == "__main__":
    main()
