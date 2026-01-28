prompt:

音乐：`./music.mp3`  
选用在频谱图上叠加的方式，生成一个动态视频，视频分辨率为1920x1080，在一个config文件中配置参数。

## 参数配置

详见 [config.yaml](config.yaml)，主要参数包括：

| 类别 | 参数 | 说明 |
|------|------|------|
| **范围** | `range.start_time` / `range.end_time` | 选择歌曲的时间范围（秒） |
| **视频** | `video.width` / `video.height` / `video.fps` | 输出视频分辨率和帧率 |
| **音频** | `audio.window_duration` / `audio.hop_duration` | 分析窗口和滑动步长 |
| **频谱** | `spectrogram.n_mels` / `spectrogram.colormap` | 梅尔频谱图参数 |
| **注意力** | `attention.layer_index` / `attention.head_reduction` | 选择可视化的注意力层和聚合方式 |
| **叠加** | `overlay.attention_weight` / `overlay.mode` | 控制叠加效果 |
| **布局** | `layout.mode` | `single`(叠加) 或 `split`(分屏) |

## 使用方法

```bash
# 安装依赖
pip install -e .

# 使用默认配置生成视频
python visualize.py

# 使用自定义配置
python visualize.py --config my_config.yaml
```