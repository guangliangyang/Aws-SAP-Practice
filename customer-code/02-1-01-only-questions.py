import pdfplumber
import re
import json
from typing import Dict, List, Any

class AWSExamQuestionParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.questions: List[Dict[str, Any]] = []

    def extract_text_from_pdf(self) -> str:
        full_text = ''
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or ''
        # 替换无效字符，并移除所有"\nMost Voted"标记
        full_text = full_text.replace('\u0000', 'f')
        full_text = full_text.replace('\nMost Voted', '')
        return full_text

    def parse_questions(self):
        text = self.extract_text_from_pdf()
        # 改进后的正则表达式，确保只匹配到正确答案行
        question_pattern = r'(Question\s*#\d+).*?(?:Topic\s*\d+)?\n(.*?)\nA[\.\:]\s*(.*?)\nB[\.\:]\s*(.*?)\nC[\.\:]\s*(.*?)\nD[\.\:]\s*(.*?)(?:\nE[\.\:]\s*(.*?))?(?:\nF[\.\:]\s*(.*?))?\nCorrect\s*Answer\s*:\s*([A-Z,\s]+)(?:\n|$)'

        matches = re.finditer(question_pattern, text, re.DOTALL)

        for match in matches:
            try:
                question_header = match.group(1)
                question_stem = match.group(2).strip()

                # 处理选项文本，去除"Most Voted"字样
                def clean_option_text(option_text):
                    return re.sub(r'\s*\[Most\s+Voted\]\s*$', '', option_text).strip()

                options = {
                    'A': clean_option_text(match.group(3)),
                    'B': clean_option_text(match.group(4)),
                    'C': clean_option_text(match.group(5)),
                    'D': clean_option_text(match.group(6)),
                }

                # 处理可选E和F选项
                if match.group(7):
                    options['E'] = clean_option_text(match.group(7))
                if match.group(8):
                    options['F'] = clean_option_text(match.group(8))

                # 处理正确答案
                correct_answer_str = match.group(9).strip()
                correct_answers = list(re.sub(r'[^A-Z]', '', correct_answer_str.upper()))

                # 验证每个答案选项是否存在
                valid_answers = [opt for opt in correct_answers if opt in options]
                if not valid_answers:
                    print(
                        f"警告: 问题 {question_header} 的正确答案选项不存在: {correct_answers} (原始: {correct_answer_str})")
                    continue

                parsed_question = {
                    'question_number': re.search(r'Question\s*#(\d+)', question_header).group(1),
                    'stem': question_stem,
                    'options': options,
                    'correct_answer': valid_answers
                }

                self.questions.append(parsed_question)

            except Exception as e:
                print(f"Error parsing question: {e}")
                continue

    def print_questions(self):
        for q in self.questions:
            print(f"Question #{q['question_number']}:")
            print(f"题干: {q['stem']}")
            for option, text in q['options'].items():
                print(f"{option}: {text}")
            print(f"Correct Answer: {', '.join(q['correct_answer'])}\n")

    def export_to_json(self, output_file='aws_exam_questions.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.questions, f, ensure_ascii=False, indent=2)

        print(f"问题数据已导出到 {output_file}")


def main():
    pdf_path = 'MLS-C01.pdf'  # 替换为您的实际PDF路径
    parser = AWSExamQuestionParser(pdf_path)
    parser.parse_questions()
    parser.print_questions()
    parser.export_to_json()


if __name__ == '__main__':
    main()