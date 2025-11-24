from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
from docx import Document
import jieba
import jieba.posseg as pseg
from collections import Counter, defaultdict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import io
import base64
import re
import numpy as np
from datetime import datetime
import random

app = Flask(__name__)
app.secret_key = 'novel_analysis_secret_key_2024'

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置中文字体
try:
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    ]

    chinese_font = None
    for path in font_paths:
        if os.path.exists(path):
            chinese_font = fm.FontProperties(fname=path)
            break

    if chinese_font is None:
        chinese_font = fm.FontProperties()
        print("警告: 未找到中文字体，图表可能无法正确显示中文")
    else:
        plt.rcParams['font.family'] = chinese_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字体配置错误: {e}")

# 关系类型定义
RELATIONSHIP_TYPES = {
    '家人': ['父亲', '母亲', '儿子', '女儿', '兄弟', '姐妹', '丈夫', '妻子', '祖父', '祖母', '叔叔', '阿姨', '侄子',
             '侄女'],
    '朋友': ['朋友', '好友', '知己', '伙伴', '同伴', '同学', '同事'],
    '敌对': ['敌人', '仇人', '对手', '竞争者', '对头'],
    '师徒': ['师父', '徒弟', '老师', '学生', '师傅', '学徒'],
    '上下级': ['上司', '下属', '领导', '部下', '主人', '仆人', '皇帝', '臣子'],
    '爱人': ['爱人', '恋人', '情侣', '夫妻', '未婚夫', '未婚妻']
}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_docx(file_path):
    """
    读取Word文档内容
    """
    try:
        doc = Document(file_path)
        full_text = []
        chapter_positions = []

        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                if is_chapter_title(text, i):
                    chapter_positions.append((text, len(full_text)))
                full_text.append(text)

        return '\n'.join(full_text), chapter_positions
    except Exception as e:
        raise Exception(f"读取Word文档失败: {str(e)}")


def is_chapter_title(text, position):
    """
    判断是否为章节标题
    """
    chapter_patterns = [
        r'^第[零一二三四五六七八九十百千\d]+[章节回]',
        r'^[上下]?篇',
        r'^卷[一二三四五六七八九十]',
        r'^引子$',
        r'^序幕$',
        r'^尾声$',
        r'^后记$',
    ]

    for pattern in chapter_patterns:
        if re.match(pattern, text):
            return True

    if len(text) < 30 and position < 10:
        return True

    return False


def clean_text(input_text):
    """清理文本"""
    cleaned = re.sub(r'\s+', ' ', input_text)
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；："\'（）《》]', '', cleaned)
    return cleaned.strip()


def analyze_word_frequency(text, top_n=30):
    """分析词频"""
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    words = jieba.lcut(cleaned_text)

    stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说',
        '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们', '他们', '她们',
        '这个', '那个', '这样', '那样', '什么', '怎么', '为什么', '因为', '所以', '但是', '然后', '已经', '可以', '应该'
    }

    filtered_words = [
        word for word in words
        if (len(word) > 1 and not word.isspace() and word not in stop_words)
    ]

    if not filtered_words:
        return []

    word_freq = Counter(filtered_words)
    total_count = sum(word_freq.values())

    most_common = word_freq.most_common(top_n)
    result = []
    for word, count in most_common:
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        result.append((word, count, percentage))

    return result


def extract_characters_with_context(text, top_n=15):
    """
    提取人物并分析上下文关系
    """
    # 使用jieba进行词性标注
    words = pseg.cut(text)

    # 提取可能的人名
    characters = []
    for word, flag in words:
        if flag == 'nr' and len(word) > 1:
            characters.append(word)
        elif flag in ['n', 'nh'] and len(word) > 1:
            if not is_not_person_name(word):
                characters.append(word)

    # 统计人物出现频率
    char_freq = Counter(characters)
    main_characters = char_freq.most_common(top_n)

    # 分析人物关系
    relationships = analyze_character_relationships_advanced(text, main_characters)

    return main_characters, relationships


def is_not_person_name(word):
    """判断是否不是人名"""
    non_person_words = {
        '时候', '地方', '东西', '事情', '问题', '工作', '公司', '学校', '国家', '城市',
        '今天', '昨天', '明天', '时间', '朋友', '老师', '学生', '医生', '警察', '老板'
    }
    return word in non_person_words


def analyze_character_relationships_advanced(text, characters):
    """
    高级人物关系分析
    通过分析人物在上下文中的互动来判断关系类型
    """
    main_chars = [char[0] for char in characters]

    # 按句子分割文本
    sentences = re.split(r'[。！？]', text)

    relationships = []

    for sentence in sentences:
        if len(sentence.strip()) < 5:  # 跳过过短的句子
            continue

        # 找出在当前句子中出现的人物
        present_chars = []
        for char in main_chars:
            if char in sentence:
                present_chars.append(char)

        # 如果句子中有两个或更多人物，分析他们的关系
        if len(present_chars) >= 2:
            relationship = analyze_sentence_relationship(sentence, present_chars)
            if relationship:
                relationships.append(relationship)

    # 合并相同的关系
    merged_relationships = merge_relationships(relationships)

    return merged_relationships


def analyze_sentence_relationship(sentence, characters):
    """
    分析句子中人物关系
    """
    # 关系关键词模式
    patterns = {
        '父子': [r'(\w+)的(父亲|爸爸|爹)', r'(\w+)的(儿子|孩子|小儿)'],
        '母女': [r'(\w+)的(母亲|妈妈|娘)', r'(\w+)的(女儿|闺女|女)'],
        '兄弟': [r'(\w+)的(哥哥|兄长|大哥)', r'(\w+)的(弟弟|兄弟|小弟)'],
        '姐妹': [r'(\w+)的(姐姐|大姐)', r'(\w+)的(妹妹|小妹)'],
        '夫妻': [r'(\w+)的(妻子|老婆|夫人)', r'(\w+)的(丈夫|老公|先生)'],
        '朋友': [r'(\w+)的(朋友|好友|伙伴)', r'(\w+)和(\w+)是(朋友|好友)'],
        '敌人': [r'(\w+)的(敌人|仇人|对头)', r'(\w+)和(\w+)是(敌人|仇人)'],
        '师徒': [r'(\w+)的(师父|师傅|老师)', r'(\w+)的(徒弟|学生|弟子)'],
        '上下级': [r'(\w+)的(上司|领导|老板)', r'(\w+)的(下属|部下|员工)']
    }

    for rel_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, sentence)
            for match in matches:
                # 确保匹配到的人物在characters列表中
                for char in characters:
                    if char in match:
                        # 找到关系中的另一个人物
                        for other_char in characters:
                            if other_char != char and other_char in sentence:
                                return {
                                    'source': char,
                                    'target': other_char,
                                    'relationship': rel_type,
                                    'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence
                                }

    # 如果没有明确的关系模式，根据上下文推断
    return infer_relationship_from_context(sentence, characters)


def infer_relationship_from_context(sentence, characters):
    """
    根据上下文推断关系
    """
    if len(characters) < 2:
        return None

    # 情感词分析
    positive_words = {'帮助', '关心', '爱护', '支持', '鼓励', '赞美', '感谢', '喜欢', '爱'}
    negative_words = {'伤害', '攻击', '批评', '讨厌', '恨', '背叛', '欺骗', '威胁'}

    sentence_words = set(jieba.lcut(sentence))

    if sentence_words & positive_words:
        relationship_type = '朋友'
    elif sentence_words & negative_words:
        relationship_type = '敌人'
    else:
        relationship_type = '相识'  # 默认关系

    return {
        'source': characters[0],
        'target': characters[1],
        'relationship': relationship_type,
        'sentence': sentence[:100] + '...' if len(sentence) > 100 else sentence
    }


def merge_relationships(relationships):
    """
    合并相同的人物关系
    """
    merged = {}

    for rel in relationships:
        if rel is None:
            continue

        key = (rel['source'], rel['target'])
        reverse_key = (rel['target'], rel['source'])

        # 检查是否已经存在相同或相反的关系
        if key in merged:
            # 如果关系类型不同，选择更具体的关系
            if merged[key]['relationship'] == '相识' and rel['relationship'] != '相识':
                merged[key] = rel
        elif reverse_key in merged:
            # 处理相反方向的关系
            existing_rel = merged[reverse_key]
            # 确保关系类型一致
            if existing_rel['relationship'] == rel['relationship']:
                merged[reverse_key]['sentence'] += f" | {rel['sentence']}"
        else:
            merged[key] = rel

    return list(merged.values())


def identify_main_character(characters, relationships):
    """
    识别主角
    基于出现频率和关系网络中心性
    """
    if not characters:
        return None

    # 构建关系图
    G = nx.Graph()

    # 添加节点（人物）
    for char, freq in characters:
        G.add_node(char, frequency=freq)

    # 添加边（关系）
    for rel in relationships:
        G.add_edge(rel['source'], rel['target'], relationship=rel['relationship'])

    # 计算度中心性（与其他节点的连接数）
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)

        # 结合频率和中心性选择主角
        best_score = -1
        main_char = None

        for char, freq in characters:
            centrality = degree_centrality.get(char, 0)
            score = freq * (1 + centrality * 10)  # 频率和中心性的加权
            if score > best_score:
                best_score = score
                main_char = char

        return main_char

    # 如果没有关系数据，返回出现频率最高的人物
    return characters[0][0]


def analyze_plot_development_advanced(text, chapter_positions, main_character):
    """
    高级情节发展分析
    """
    # 如果没有检测到章节，将文本分为5个等分
    if not chapter_positions:
        segments = split_text_into_segments(text, 5)
        chapter_data = [{"title": f"第{i + 1}部分", "content": seg, "position": i} for i, seg in enumerate(segments)]
    else:
        # 根据章节位置分割文本
        chapter_data = []
        for i, (title, pos) in enumerate(chapter_positions):
            if i < len(chapter_positions) - 1:
                next_pos = chapter_positions[i + 1][1]
                content = "\n".join(text.split('\n')[pos:next_pos])
            else:
                content = "\n".join(text.split('\n')[pos:])

            chapter_data.append({
                "title": title,
                "content": content,
                "position": i
            })

    # 分析每个章节的情节
    for chapter in chapter_data:
        content = chapter["content"]

        # 分析主角行为和心理
        character_analysis = analyze_character_actions(content, main_character)
        chapter.update(character_analysis)

        # 分析情节作用
        plot_function = analyze_plot_function(content, chapter["position"], len(chapter_data))
        chapter["plot_function"] = plot_function

        # 提取关键事件
        key_events = extract_key_events(content, main_character)
        chapter["key_events"] = key_events

    return chapter_data


def split_text_into_segments(text, n_segments=5):
    """将文本分割为n个等分"""
    paragraphs = [p for p in text.split('\n') if p.strip()]
    segment_size = max(1, len(paragraphs) // n_segments)

    segments = []
    for i in range(n_segments):
        start = i * segment_size
        if i == n_segments - 1:
            end = len(paragraphs)
        else:
            end = (i + 1) * segment_size

        segment_text = "\n".join(paragraphs[start:end])
        segments.append(segment_text)

    return segments


def analyze_character_actions(text, main_character):
    """
    分析主角行为和心理状态
    """
    if not main_character:
        return {
            "location": "未知",
            "action": "未知",
            "mental_state": "未知",
            "time": "未知"
        }

    # 提取包含主角的句子
    sentences = re.split(r'[。！？]', text)
    character_sentences = [s for s in sentences if main_character in s]

    if not character_sentences:
        return {
            "location": "未知",
            "action": "未知",
            "mental_state": "未知",
            "time": "未知"
        }

    # 分析地点
    location = extract_location(character_sentences)

    # 分析时间
    time_info = extract_time(character_sentences)

    # 分析行为
    action = extract_action(character_sentences, main_character)

    # 分析心理状态
    mental_state = analyze_mental_state(character_sentences)

    return {
        "location": location,
        "action": action,
        "mental_state": mental_state,
        "time": time_info
    }


def extract_location(sentences):
    """提取地点信息"""
    location_keywords = ['在', '到', '去', '来到', '走进', '离开', '从', '向']
    location_indicators = ['家', '学校', '公司', '房间', '街道', '城市', '山', '河', '海']

    for sentence in sentences:
        for keyword in location_keywords:
            if keyword in sentence:
                # 提取关键词后面的内容作为地点
                idx = sentence.find(keyword)
                location_text = sentence[idx:min(idx + 20, len(sentence))]
                return location_text

    # 如果没有明确地点，返回默认值
    return "故事发生地"


def extract_time(sentences):
    """提取时间信息"""
    time_keywords = ['今天', '明天', '昨天', '早上', '中午', '晚上', '春天', '夏天', '秋天', '冬天']

    for sentence in sentences:
        for keyword in time_keywords:
            if keyword in sentence:
                return keyword

    return "某个时间"


def extract_action(sentences, main_character):
    """提取主角行为"""
    action_verbs = ['说', '做', '走', '跑', '看', '听', '想', '感觉', '决定', '开始', '结束']

    for sentence in sentences:
        if main_character in sentence:
            # 提取主角后面的动词作为行为
            idx = sentence.find(main_character)
            action_part = sentence[idx:min(idx + 15, len(sentence))]
            return action_part

    return "进行某些活动"


def analyze_mental_state(sentences):
    """分析心理状态"""
    positive_words = {'高兴', '快乐', '开心', '幸福', '喜悦', '愉快', '兴奋', '激动', '满意'}
    negative_words = {'悲伤', '难过', '痛苦', '伤心', '绝望', '失望', '愤怒', '生气', '害怕', '恐惧'}

    all_words = []
    for sentence in sentences:
        all_words.extend(jieba.lcut(sentence))

    positive_count = sum(1 for word in all_words if word in positive_words)
    negative_count = sum(1 for word in all_words if word in negative_words)

    if positive_count > negative_count:
        return "积极"
    elif negative_count > positive_count:
        return "消极"
    else:
        return "平静"


def analyze_plot_function(text, position, total_chapters):
    """
    分析情节作用
    """
    # 根据章节位置判断情节作用
    if position == 0:
        return "开端：介绍背景和人物，设置故事的基本情境"
    elif position < total_chapters * 0.3:
        return "发展：推动故事前进，建立冲突和人物关系"
    elif position < total_chapters * 0.7:
        return "高潮：故事的关键转折点，冲突达到顶峰"
    elif position == total_chapters - 1:
        return "结局：解决冲突，完成故事弧线"
    else:
        return "过渡：连接不同情节部分，维持故事节奏"


def extract_key_events(text, main_character):
    """
    提取关键事件
    """
    sentences = re.split(r'[。！？]', text)
    key_events = []

    # 关键事件指示词
    event_indicators = ['突然', '意外', '决定', '发现', '遇见', '离开', '开始', '结束', '改变']

    for sentence in sentences:
        if main_character and main_character in sentence:
            for indicator in event_indicators:
                if indicator in sentence:
                    key_events.append(sentence.strip())
                    break

    return key_events[:3]  # 返回最多3个关键事件


def calculate_text_stats(text):
    """计算文本统计信息"""
    total_chars = len(text)

    paragraphs = [p for p in text.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)

    words = jieba.lcut(clean_text(text))
    filtered_words = [word for word in words if len(word) > 1 and not word.isspace()]
    total_words = len(filtered_words)
    unique_words = len(set(filtered_words))

    reading_time = max(1, total_chars // 300)

    return {
        'total_chars': total_chars,
        'paragraph_count': paragraph_count,
        'total_words': total_words,
        'unique_words': unique_words,
        'reading_time': reading_time
    }


def generate_word_freq_chart(word_freq):
    """生成词频统计图表"""
    if not word_freq:
        return None

    words = [item[0] for item in word_freq]
    frequencies = [item[1] for item in word_freq]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(words, frequencies, color='skyblue')

    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_width() + max(frequencies) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{freq}', ha='left', va='center', fontsize=10)

    plt.xlabel('出现次数', fontsize=12)
    plt.title('高频词汇统计 TOP 30', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


def generate_relationship_network(characters, relationships, main_character):
    """生成人物关系网络图"""
    if not characters or not relationships:
        return None

    # 创建图形
    plt.figure(figsize=(14, 10))
    G = nx.Graph()

    # 添加节点
    for char, freq in characters:
        G.add_node(char, frequency=freq)

    # 添加边
    for rel in relationships:
        G.add_edge(rel['source'], rel['target'], relationship=rel['relationship'])

    # 设置节点位置（使用spring布局）
    pos = nx.spring_layout(G, k=3, iterations=50)

    # 绘制节点 - 主角用不同颜色
    node_colors = []
    for node in G.nodes():
        if node == main_character:
            node_colors.append('red')  # 主角红色
        else:
            node_colors.append('lightblue')  # 配角浅蓝色

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, alpha=0.9)

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family=chinese_font.get_name())

    # 绘制边和关系标签
    for rel in relationships:
        source, target = rel['source'], rel['target']
        if G.has_edge(source, target):
            # 绘制边
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], width=2, alpha=0.7)

            # 添加关系标签
            x = (pos[source][0] + pos[target][0]) / 2
            y = (pos[source][1] + pos[target][1]) / 2
            plt.text(x, y, rel['relationship'], fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                     horizontalalignment='center')

    plt.title("人物关系网络图", fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


def generate_plot_chart(plot_data):
    """生成情节发展图表"""
    if not plot_data:
        return None

    # 提取情感分数
    emotions = [chapter.get("emotion_score", 0) for chapter in plot_data]
    titles = [chapter["title"] for chapter in plot_data]

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 绘制情感曲线
    x = range(len(emotions))
    plt.plot(x, emotions, marker='o', linewidth=2, markersize=8, color='steelblue')

    # 填充情感区域
    plt.fill_between(x, emotions, alpha=0.3, color='lightblue')

    # 设置坐标轴
    plt.xticks(x, titles, rotation=45)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    plt.xlabel('章节', fontsize=12)
    plt.ylabel('情感分数', fontsize=12)
    plt.title('情节情感发展', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        if 'file' not in request.files:
            flash('请选择要上传的文件')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('请选择要上传的文件')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            text_content, chapter_positions = read_docx(file_path)

            if not text_content.strip():
                flash('文档内容为空，请上传包含内容的Word文档')
                return redirect(request.url)

            # 计算文本统计
            stats = calculate_text_stats(text_content)

            # 分析词频
            word_freq = analyze_word_frequency(text_content)

            # 分析人物和关系
            characters, relationships = extract_characters_with_context(text_content)

            # 识别主角
            main_character = identify_main_character(characters, relationships)

            # 分析情节发展
            plot_development = analyze_plot_development_advanced(text_content, chapter_positions, main_character)

            # 生成图表
            word_chart = generate_word_freq_chart(word_freq)
            relationship_chart = generate_relationship_network(characters, relationships, main_character)
            plot_chart = generate_plot_chart(plot_development)

            # 文本预览
            preview_text = text_content[:500] + '...' if len(text_content) > 500 else text_content
            preview_percentage = min(100, int((500 / len(text_content)) * 100)) if len(text_content) > 500 else 100

            # 返回分析结果
            return render_template('index.html',
                                   text_preview=preview_text,
                                   word_freq=word_freq,
                                   characters=characters,
                                   relationships=relationships,
                                   plot_development=plot_development,
                                   main_character=main_character,
                                   word_chart=word_chart,
                                   relationship_chart=relationship_chart,
                                   plot_chart=plot_chart,
                                   filename=filename,
                                   total_words=stats['total_words'],
                                   unique_words=stats['unique_words'],
                                   text_length=stats['total_chars'],
                                   paragraph_count=stats['paragraph_count'],
                                   reading_time=stats['reading_time'],
                                   preview_percentage=preview_percentage,
                                   analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        else:
            flash('只支持 .docx 格式的Word文档')
            return redirect(request.url)

    except Exception as e:
        flash(f'处理文件时出错: {str(e)}')
        return redirect(request.url)


@app.errorhandler(413)
def too_large(error):
    """文件过大错误处理"""
    flash('文件大小超过限制（最大16MB）')
    return redirect(request.url)


if __name__ == '__main__':
    jieba.initialize()

    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("=" * 50)
    print("小说内容分析系统启动成功！")
    print("功能：词频分析、人物关系网络、情节发展分析")
    print("访问地址: http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)