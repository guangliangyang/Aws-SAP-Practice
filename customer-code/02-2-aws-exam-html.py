import json
from pathlib import Path


def json_to_html(input_json='aws_exam_categorized_questions.json', output_html='aws_exam_questions.html'):
    # 读取JSON数据
    if not Path(input_json).exists():
        print(f"错误：找不到输入文件 {input_json}")
        return

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建HTML内容
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AWS认证考试题库</title>
    <style>
        :root {{
            --primary-color: #2d3748;
            --secondary-color: #4a5568;
            --accent-color: #4299e1;
            --correct-color: #38a169;
            --bg-color: #f7fafc;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--primary-color);
            background-color: var(--bg-color);
            display: flex;
            min-height: 100vh;
        }}
        #sidebar {{
            width: 280px;
            background: white;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            padding: 20px;
            overflow-y: auto;
            position: fixed;
            height: 100vh;
        }}
        #content {{
            flex: 1;
            margin-left: 280px;
            padding: 30px;
        }}
        .logo {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-color);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        .search-box {{
            margin-bottom: 20px;
        }}
        #searchInput {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .category-list {{
            list-style: none;
        }}
        .main-category {{
            font-weight: bold;
            margin-top: 15px;
            color: var(--secondary-color);
            cursor: pointer;
            padding: 5px 0;
        }}
        .sub-category {{
            margin-left: 15px;
            padding: 5px 0;
            cursor: pointer;
            color: var(--primary-color);
        }}
        .sub-category:hover, .main-category:hover {{
            color: var(--accent-color);
        }}
        .active-category {{
            color: var(--accent-color);
            font-weight: bold;
        }}
        .question {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .question-number {{
            font-weight: bold;
            font-size: 1.1rem;
            color: var(--accent-color);
            margin-bottom: 10px;
        }}
        .question-stem {{
            margin-bottom: 15px;
        }}
        .option {{
            margin-bottom: 8px;
            padding-left: 20px;
            position: relative;
        }}
        .option::before {{
            content: attr(data-option) ":";
            font-weight: bold;
            position: absolute;
            left: 0;
        }}
        .correct-answer {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f0fff4;
            border-left: 3px solid var(--correct-color);
            color: var(--correct-color);
            font-weight: bold;
        }}
        .correct-option {{
            display: block;
            margin-bottom: 5px;
        }}
        .hidden {{
            display: none;
        }}
        .category-header {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--secondary-color);
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }}
        @media (max-width: 768px) {{
            body {{
                flex-direction: column;
            }}
            #sidebar {{
                width: 100%;
                height: auto;
                position: relative;
            }}
            #content {{
                margin-left: 0;
            }}
        }}
    </style>
</head>
<body>
    <aside id="sidebar">
        <div class="logo">AWS认证题库</div>
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="搜索题目...">
        </div>
        <ul class="category-list" id="categoryList">
"""

    # 生成分类导航
    categories = sorted(data.keys())
    current_main_category = ""

    for category in categories:
        main_category, sub_category = category.split(" - ")
        if main_category != current_main_category:
            html += f'<li class="main-category" data-main-category="{main_category}">{main_category}</li>\n'
            current_main_category = main_category
        html += f'<li class="sub-category hidden" data-category="{category}" data-main-category="{main_category}">{sub_category}</li>\n'

    html += """
        </ul>
    </aside>

    <main id="content">
        <div id="questionsContainer">
"""

    # 生成题目内容
    for category, questions in data.items():
        html += f'<div class="category-questions hidden" data-category="{category}">\n'
        html += f'<h2 class="category-header">{category}</h2>\n'

        for q in questions:
            options_html = ""
            for opt, text in q["options"].items():
                options_html += f'<div class="option" data-option="{opt}">{text}</div>\n'

            # 修改这里：每个正确答案单独一行
            correct_answers_html = ""
            for opt in q["correct_answer"]:
                if opt in q["correct_answer_content"]:
                    correct_answers_html += f'<span class="correct-option">{opt}: {q["correct_answer_content"][opt]}</span>\n'
                else:
                    correct_answers_html += f'<span class="correct-option">{opt}</span>\n'

            html += f"""
            <div class="question" data-number="{q['question_number']}" data-text="{q['stem'].lower()} {' '.join(q['options'].values()).lower()}">
                <div class="question-number">题目 #{q['question_number']}</div>
                <div class="question-stem">{q['stem']}</div>
                <div class="options">
                    {options_html}
                </div>
                <div class="correct-answer">
                    <strong>正确答案:</strong><br>
                    {correct_answers_html}
                </div>
            </div>
            """
        html += "</div>\n"

    html += """
        </div>
    </main>

    <script>
        // 主分类点击事件 - 显示/隐藏子分类
        document.querySelectorAll('.main-category').forEach(item => {
            item.addEventListener('click', function() {
                const mainCategory = this.getAttribute('data-main-category');
                const subCategories = document.querySelectorAll(`.sub-category[data-main-category="${mainCategory}"]`);

                subCategories.forEach(subCat => {
                    subCat.classList.toggle('hidden');
                });
            });
        });

        // 子分类点击事件 - 显示对应题目
        document.querySelectorAll('.sub-category').forEach(item => {
            item.addEventListener('click', function() {
                // 更新活动状态
                document.querySelectorAll('.sub-category').forEach(cat => {
                    cat.classList.remove('active-category');
                });
                this.classList.add('active-category');

                // 隐藏所有题目
                document.querySelectorAll('.category-questions').forEach(q => {
                    q.classList.add('hidden');
                });

                // 显示选中分类的题目
                const category = this.getAttribute('data-category');
                document.querySelector(`.category-questions[data-category="${category}"]`).classList.remove('hidden');

                // 滚动到顶部
                window.scrollTo(0, 0);
            });
        });

        // 默认显示第一个分类的第一个子分类
        const firstMainCategory = document.querySelector('.main-category');
        if (firstMainCategory) {
            firstMainCategory.click();
            const firstSubCategory = document.querySelector('.sub-category:not(.hidden)');
            if (firstSubCategory) {
                firstSubCategory.click();
            }
        }

        // 搜索功能
        document.getElementById('searchInput').addEventListener('input', function() {
            const searchText = this.value.toLowerCase().trim();

            // 如果搜索框为空，显示当前分类的所有题目
            if (searchText === '') {
                const activeCategory = document.querySelector('.category-questions:not(.hidden)');
                if (activeCategory) {
                    const questions = activeCategory.querySelectorAll('.question');
                    questions.forEach(q => q.classList.remove('hidden'));
                }
                return;
            }

            // 在当前显示的题目中搜索
            const activeQuestions = document.querySelectorAll('.category-questions:not(.hidden) .question');
            let hasResults = false;

            activeQuestions.forEach(q => {
                const questionText = q.getAttribute('data-text');
                if (questionText.includes(searchText)) {
                    q.classList.remove('hidden');
                    hasResults = true;
                } else {
                    q.classList.add('hidden');
                }
            });

            // 如果没有结果，提示用户
            if (!hasResults) {
                // 可以在这里添加"没有找到结果"的提示
            }
        });
    </script>
</body>
</html>
"""

    # 写入HTML文件
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"成功生成HTML文件: {output_html}")


if __name__ == '__main__':
    json_to_html()