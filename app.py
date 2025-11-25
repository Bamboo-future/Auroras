from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from docx import Document
import jieba
import jieba.posseg as pseg
from collections import Counter, defaultdict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import re
import numpy as np
from datetime import datetime
import networkx as nx
import math
import tempfile

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'novel_analysis_secret_key_2024_commercial')

# 配置
UPLOAD_FOLDER = 'tmp_uploads'  # 使用临时目录
ALLOWED_EXTENSIONS = {'docx'}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 关系模式识别词典
RELATIONSHIP_PATTERNS = {
    '父子': ['父亲', '爸爸', '爹', '老爸', '父'],
    '母子': ['母亲', '妈妈', '娘', '老妈', '母'],
    '夫妻': ['丈夫', '妻子', '老婆', '老公', '夫君', '夫人', '配偶'],
    '兄弟姐妹': ['哥哥', '弟弟', '姐姐', '妹妹', '兄', '弟', '姐', '妹', '兄弟', '姐妹'],
    '朋友': ['朋友', '好友', '伙伴', '同伴', '哥们', '闺蜜'],
    '师徒': ['师父', '师傅', '徒弟', '弟子', '学生', '老师'],
    '敌人': ['敌人', '仇人', '对手', '仇敌', '对头'],
    '上下级': ['上司', '下属', '领导', '部下', '老板', '员工']
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_docx(file_path):
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
    chapter_patterns = [
        r'^第[零一二三四五六七八九十百千\d]+[章节回]',
        r'^[上下]?篇',
        r'^卷[一二三四五六七八九十]',
    ]

    for pattern in chapter_patterns:
        if re.match(pattern, text):
            return True

    if len(text) < 30 and position < 10:
        return True

    return False


def clean_text(input_text):
    cleaned = re.sub(r'\s+', ' ', input_text)
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；："\'（）《》]', '', cleaned)
    return cleaned.strip()


def analyze_word_frequency(text, top_n=30):
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


def extract_characters_advanced(text, top_n=20):
    words = pseg.cut(text)
    potential_chars = []

    for word, flag in words:
        if flag == 'nr' and len(word) > 1:
            potential_chars.append(word)

    char_stats = defaultdict(lambda: {'count': 0, 'positions': [], 'dialogues': 0})

    paragraphs = [p for p in text.split('\n') if p.strip()]
    total_paragraphs = len(paragraphs)

    for i, para in enumerate(paragraphs):
        for char in potential_chars:
            if char in para:
                char_stats[char]['count'] += 1
                char_stats[char]['positions'].append(i)

                if '说' in para or '道' in para or '问' in para or '答' in para:
                    char_stats[char]['dialogues'] += 1

    char_scores = []
    for char, stats in char_stats.items():
        if stats['count'] < 2:
            continue

        positions = stats['positions']
        if len(positions) > 1:
            position_variance = np.var(positions)
            position_score = 1 / (1 + position_variance / 1000)
        else:
            position_score = 0.5

        dialogue_score = stats['dialogues'] / max(1, stats['count'])
        total_score = stats['count'] * position_score * (1 + dialogue_score)

        char_scores.append((char, total_score, stats['count']))

    char_scores.sort(key=lambda x: x[1], reverse=True)

    if char_scores:
        main_count = min(3, len(char_scores))
        main_chars = [(char, score, count) for char, score, count in char_scores[:main_count]]
        supporting_chars = [(char, score, count) for char, score, count in char_scores[main_count:top_n]]

        return {
            'main_characters': main_chars,
            'supporting_characters': supporting_chars
        }

    return {'main_characters': [], 'supporting_characters': []}


def analyze_character_relationships_advanced(text, characters_data):
    main_chars = [char[0] for char in characters_data['main_characters']]
    all_chars = main_chars + [char[0] for char in characters_data['supporting_characters']]

    G = nx.Graph()

    for char in all_chars:
        G.add_node(char, type='main' if char in main_chars else 'supporting')

    relationships = []
    paragraphs = [p for p in text.split('\n') if p.strip()]

    for para in paragraphs:
        present_chars = []
        for char in all_chars:
            if char in para:
                present_chars.append(char)

        for i in range(len(present_chars)):
            for j in range(i + 1, len(present_chars)):
                char1, char2 = present_chars[i], present_chars[j]
                relation_type = analyze_relationship_type(para, char1, char2)

                if relation_type:
                    if G.has_edge(char1, char2):
                        G[char1][char2]['weight'] += 1
                        G[char1][char2]['types'].add(relation_type)
                    else:
                        G.add_edge(char1, char2, weight=1, types={relation_type})

    relationship_data = []
    for edge in G.edges(data=True):
        char1, char2, data = edge
        weight = data['weight']
        types = list(data['types'])
        main_type = types[0] if types else '相关'

        relationship_data.append({
            'source': char1,
            'target': char2,
            'type': main_type,
            'strength': weight
        })

    relationship_chart = generate_relationship_chart(G, characters_data)

    return relationship_data, relationship_chart


def analyze_relationship_type(paragraph, char1, char2):
    for relation_type, keywords in RELATIONSHIP_PATTERNS.items():
        for keyword in keywords:
            if keyword in paragraph:
                char1_pos = paragraph.find(char1)
                char2_pos = paragraph.find(char2)
                keyword_pos = paragraph.find(keyword)

                if char1_pos != -1 and char2_pos != -1 and keyword_pos != -1:
                    min_pos = min(char1_pos, char2_pos)
                    max_pos = max(char1_pos, char2_pos)

                    if min_pos <= keyword_pos <= max_pos:
                        return relation_type

                    if abs(keyword_pos - char1_pos) < 20 or abs(keyword_pos - char2_pos) < 20:
                        return relation_type

    return None


def generate_relationship_chart(G, characters_data):
    if len(G.nodes()) == 0:
        return None

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G, k=2, iterations=30)

    for edge in G.edges(data=True):
        source, target, data = edge
        weight = data['weight']
        edge_width = max(1, min(3, weight / 2))

        nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                               width=edge_width, alpha=0.7, edge_color='gray')

    main_chars = [char[0] for char in characters_data['main_characters']]

    node_colors = []
    for node in G.nodes():
        if node in main_chars:
            node_colors.append('#ff6b6b')
        else:
            node_colors.append('#4ecdc4')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='SimHei')

    plt.title("人物关系网络图", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=80, bbox_inches='tight', facecolor='white')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


def analyze_plot_development_advanced(text, chapter_positions, main_characters):
    if not chapter_positions:
        segments = split_text_into_segments(text, 4)
        chapter_data = [{"title": f"第{i + 1}部分", "content": seg, "position": i} for i, seg in enumerate(segments)]
    else:
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

    plot_analysis = []
    for i, chapter in enumerate(chapter_data):
        content = chapter["content"]

        plot_stage = determine_plot_stage(i, len(chapter_data))
        character_analysis = analyze_character_activities(content, main_characters)
        plot_function = analyze_plot_function(content, i, len(chapter_data))

        plot_analysis.append({
            "title": chapter["title"],
            "stage": plot_stage,
            "character_analysis": character_analysis,
            "plot_function": plot_function,
            "emotional_intensity": calculate_emotion_score(content),
            "key_events": extract_key_events(content, main_characters)
        })

    return plot_analysis


def determine_plot_stage(chapter_index, total_chapters):
    if chapter_index == 0:
        return "开端"
    elif chapter_index == total_chapters - 1:
        return "结局"
    elif chapter_index < total_chapters * 0.3:
        return "发展"
    elif chapter_index < total_chapters * 0.7:
        return "高潮"
    else:
        return "收尾"


def analyze_character_activities(content, main_characters):
    analysis = []
    main_chars = [char[0] for char in main_characters]

    for char in main_chars:
        if char in content:
            sentences = [s for s in re.split(r'[。！？]', content) if char in s]

            if sentences:
                sample_sentence = sentences[0]

                activity = {
                    "character": char,
                    "location": extract_location(sample_sentence),
                    "time": extract_time(sample_sentence),
                    "action": extract_main_action(sample_sentence, char),
                    "psychological_state": analyze_psychological_state(sample_sentence)
                }
                analysis.append(activity)

    return analysis


def extract_location(sentence):
    location_keywords = ['在', '于', '到', '去', '来到', '进入']
    for keyword in location_keywords:
        if keyword in sentence:
            parts = sentence.split(keyword)
            if len(parts) > 1:
                return parts[1][:20] + '...'
    return "未知地点"


def extract_time(sentence):
    time_patterns = [
        r'(\d+)年',
        r'(\d+)月',
        r'(\d+)日',
        r'([早晚午夜清晨]+)',
        r'(春天|夏天|秋天|冬天)'
    ]

    for pattern in time_patterns:
        match = re.search(pattern, sentence)
        if match:
            return match.group()

    return "未知时间"


def extract_main_action(sentence, character):
    action_verbs = ['说', '做', '走', '跑', '看', '听', '想', '感觉', '决定']

    for verb in action_verbs:
        if verb in sentence and character in sentence:
            return f"{character}{verb}"

    return f"{character}活动"


def analyze_psychological_state(sentence):
    positive_words = {'高兴', '开心', '快乐', '幸福', '满意', '兴奋', '激动'}
    negative_words = {'悲伤', '难过', '痛苦', '伤心', '愤怒', '生气', '失望', '害怕'}

    words = jieba.lcut(sentence)

    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    if positive_count > negative_count:
        return "积极"
    elif negative_count > positive_count:
        return "消极"
    else:
        return "中性"


def analyze_plot_function(content, chapter_index, total_chapters):
    functions = []

    if detect_suspense_elements(content):
        functions.append("设置悬念")

    if chapter_index == 0:
        functions.append("引入背景")
    elif chapter_index < total_chapters * 0.3:
        functions.append("推动发展")
    elif chapter_index < total_chapters * 0.7:
        functions.append("构建冲突")
    else:
        functions.append("收束情节")

    return functions if functions else ["推进故事"]


def detect_suspense_elements(content):
    suspense_indicators = ['突然', '忽然', '没想到', '意外', '惊奇', '竟然', '居然']

    for indicator in suspense_indicators:
        if indicator in content:
            return True

    return False


def extract_key_events(content, main_characters):
    main_chars = [char[0] for char in main_characters]

    events = []
    sentences = [s.strip() for s in re.split(r'[。！？]', content) if s.strip()]

    for sentence in sentences:
        has_main_char = any(char in sentence for char in main_chars)
        has_important_verb = any(verb in sentence for verb in ['发现', '决定', '开始', '结束', '遇见', '离开'])

        if has_main_char and has_important_verb and len(sentence) > 10:
            events.append(sentence[:50] + '...')

    return events[:2]


def split_text_into_segments(text, n_segments=4):
    paragraphs = [p for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return [text]

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


def calculate_emotion_score(text):
    positive_words = {
        '高兴', '快乐', '开心', '幸福', '喜悦', '愉快', '兴奋', '激动', '满意', '喜欢'
    }

    negative_words = {
        '悲伤', '难过', '痛苦', '伤心', '绝望', '失望', '愤怒', '生气', '讨厌', '恨'
    }

    words = jieba.lcut(text)
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    total_emotion_words = positive_count + negative_count
    if total_emotion_words == 0:
        return 0

    emotion_score = (positive_count - negative_count) / total_emotion_words
    return round(emotion_score, 2)


def calculate_text_stats(text):
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
    if not word_freq:
        return None

    words = [item[0] for item in word_freq[:20]]  # 只显示前20个
    frequencies = [item[1] for item in word_freq[:20]]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(words, frequencies, color='#74b9ff')

    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_width() + max(frequencies) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{freq}', ha='left', va='center', fontsize=8)

    plt.xlabel('出现次数', fontsize=10)
    plt.title('高频词汇统计 TOP 20', fontsize=12, pad=15)
    plt.grid(axis='x', alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=80, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


def generate_plot_development_chart(plot_data):
    if not plot_data:
        return None

    stages = [chapter["stage"] for chapter in plot_data]
    emotions = [chapter["emotional_intensity"] for chapter in plot_data]
    titles = [chapter["title"] for chapter in plot_data]

    plt.figure(figsize=(10, 5))

    x = range(len(emotions))
    plt.plot(x, emotions, marker='o', linewidth=2, markersize=6, color='#6c5ce7')
    plt.fill_between(x, emotions, alpha=0.3, color='#a29bfe')

    stage_colors = {'开端': '#00b894', '发展': '#0984e3', '高潮': '#e17055', '收尾': '#fdcb6e', '结局': '#6c5ce7'}

    for i, (stage, emotion) in enumerate(zip(stages, emotions)):
        color = stage_colors.get(stage, 'gray')
        plt.text(i, emotion + 0.05, stage, ha='center', va='bottom',
                 fontsize=8, color=color, fontweight='bold')

    plt.xticks(x, titles, rotation=45, fontsize=8)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('章节', fontsize=10)
    plt.ylabel('情感分数', fontsize=10)
    plt.title('情节发展与情感变化', fontsize=12, pad=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=80, bbox_inches='tight')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return f"data:image/png;base64,{chart_url}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            flash('请选择要上传的文件')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('请选择要上传的文件')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # 使用临时文件
            filename = secure_filename(file.filename)
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)

            try:
                text_content, chapter_positions = read_docx(file_path)

                if not text_content.strip():
                    flash('文档内容为空，请上传包含内容的Word文档')
                    return redirect(request.url)

                stats = calculate_text_stats(text_content)
                word_freq = analyze_word_frequency(text_content)
                characters_data = extract_characters_advanced(text_content)
                relationships, relationship_chart = analyze_character_relationships_advanced(text_content,
                                                                                             characters_data)
                plot_development = analyze_plot_development_advanced(
                    text_content, chapter_positions, characters_data['main_characters']
                )

                word_chart = generate_word_freq_chart(word_freq)
                plot_chart = generate_plot_development_chart(plot_development)

                preview_text = text_content[:300] + '...' if len(text_content) > 300 else text_content
                preview_percentage = min(100, int((300 / len(text_content)) * 100)) if len(text_content) > 300 else 100

                return render_template('index.html',
                                       text_preview=preview_text,
                                       word_freq=word_freq,
                                       characters_data=characters_data,
                                       relationships=relationships,
                                       relationship_chart=relationship_chart,
                                       plot_development=plot_development,
                                       word_chart=word_chart,
                                       plot_chart=plot_chart,
                                       filename=file.filename,
                                       total_words=stats['total_words'],
                                       unique_words=stats['unique_words'],
                                       text_length=stats['total_chars'],
                                       paragraph_count=stats['paragraph_count'],
                                       reading_time=stats['reading_time'],
                                       preview_percentage=preview_percentage,
                                       analysis_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            finally:
                # 清理临时文件
                if os.path.exists(file_path):
                    os.unlink(file_path)

        else:
            flash('只支持 .docx 格式的Word文档')
            return redirect(request.url)

    except Exception as e:
        flash(f'处理文件时出错: {str(e)}')
        return redirect(request.url)


@app.errorhandler(413)
def too_large(error):
    flash('文件大小超过限制（最大16MB）')
    return redirect(request.url)


if __name__ == '__main__':
    jieba.initialize()

    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("=" * 50)
    print("小说内容分析系统启动成功！")
    print(f"访问地址: http://localhost:{port}")
    print("=" * 50)

    app.run(debug=debug, host='0.0.0.0', port=port)