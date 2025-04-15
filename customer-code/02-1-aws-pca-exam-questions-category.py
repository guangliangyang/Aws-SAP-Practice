import pdfplumber
import re
import json
from typing import Dict, List, Any

# 分类关键词体系
CLASSIFICATION_KEYWORDS = {
    "Compute": {
        "EC2": ["ec2", "instance", "ami", "autoscaling", "launch template"],
        "容器": ["ecs", "eks", "fargate", "docker", "container"],
        "无服务器": ["lambda", "serverless", "step functions", "eventbridge"],
        "批处理": ["batch", "parallelcluster"]
    },
    "Storage": {
        "对象存储": ["s3", "bucket", "glacier", "lifecycle"],
        "块存储": ["ebs", "volume", "snapshot", "io1", "gp3"],
        "文件存储": ["efs", "fsx", "lustre", "nfs"],
        "迁移工具": ["datasync", "storage gateway", "snowball"]
    },
    "Database": {
        "关系型": ["rds", "aurora", "mysql", "postgresql", "oracle"],
        "NoSQL": ["dynamodb", "documentdb", "keyspace", "ttl"],
        "缓存": ["elasticache", "memcached", "redis"],
        "数据分析": ["redshift", "timestream"]
    },
    "Networking": {
        "VPC": ["vpc", "subnet", "nacl", "security group"],
        "混合网络": ["direct connect", "vpn", "bgp"],
        "CDN": ["cloudfront", "edge location", "distribution"],
        "服务通信": ["privatelink", "vpc endpoint", "nlb"]
    },
    "Security": {
        "IAM": ["iam", "role", "policy", "sts", "permission"],
        "加密": ["kms", "encryption", "hsm", "ssl"],
        "合规": ["config", "guardduty", "inspector", "hipaa"],
        "防护": ["waf", "shield", "firewall"]
    },
    "Management": {
        "监控": ["cloudwatch", "logs", "alarm", "metrics"],
        "部署": ["cloudformation", "cdk", "sam", "template"],
        "成本": ["cost explorer", "budget", "savings plan"],
        "运维": ["systems manager", "patch manager", "opscenter"]
    },
    "BigData": {
        "流处理": ["kinesis", "firehose", "msk"],
        "分析": ["athena", "glue", "quicksight"],
        "数据湖": ["lake formation", "data catalog", "parquet"]
    },
    "ML/AI": {
        "基础服务": ["sagemaker", "notebook", "training"],
        "AI服务": ["rekognition", "lex", "polly", "transcribe"]
    }
}


class AWSExamQuestionParser:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.questions: List[Dict[str, Any]] = []
        self.categorized_questions: Dict[str, List[Dict[str, Any]]] = {}

    def extract_text_from_pdf(self) -> str:
        full_text = ''
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or ''
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
                    'number': re.search(r'Question\s*#(\d+)', question_header).group(1),
                    'stem': question_stem,
                    'options': options,
                    'correct_answer': valid_answers,
                    'correct_answer_content': {opt: options[opt] for opt in valid_answers},
                    'categories': []
                }

                self.questions.append(parsed_question)
                self._categorize_question(parsed_question)

            except Exception as e:
                print(f"Error parsing question: {e}")
                continue

    def _categorize_question(self, question: Dict[str, Any]):
        # 将问题文本转为小写以便匹配
        full_text = (question['stem'] + ' ' +
                     ' '.join(question['options'].values()) + ' ' +
                     ' '.join(question['correct_answer_content'].values())).lower()

        # 遍历分类关键词
        for main_category, sub_categories in CLASSIFICATION_KEYWORDS.items():
            for sub_category, keywords in sub_categories.items():
                # 检查是否包含关键词
                if any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', full_text) for keyword in keywords):
                    category_key = f"{main_category} - {sub_category}"
                    if category_key not in self.categorized_questions:
                        self.categorized_questions[category_key] = []

                    # 避免重复添加
                    if not any(q['number'] == question['number'] for q in self.categorized_questions[category_key]):
                        self.categorized_questions[category_key].append(question)
                        question['categories'].append(category_key)

    def _remove_duplicate_questions(self):
        # 在每个类别下移除重复题目（保持原有逻辑）
        for category in self.categorized_questions:
            unique_questions = []
            seen_numbers = set()
            for question in self.categorized_questions[category]:
                if question['number'] not in seen_numbers:
                    unique_questions.append(question)
                    seen_numbers.add(question['number'])
            self.categorized_questions[category] = unique_questions

    def print_categorized_questions(self):
        for category, questions in self.categorized_questions.items():
            print(f"\n--- {category} 分类 ---")
            print(f"总题数: {len(questions)}")
            print("-" * 50)

            for q in questions:
                print(f"Question #{q['number']}:")
                print(f"题干: {q['stem']}")
                for option, text in q['options'].items():
                    print(f"{option}: {text}")
                print(f"Correct Answer: {', '.join(q['correct_answer'])}")
                for opt, content in q['correct_answer_content'].items():
                    print(f"{opt} 选项内容: {content}")
                print(f"Categories: {', '.join(q['categories'])}\n")

    def export_to_json(self, output_file='aws_exam_categorized_questions.json'):
        export_data = {}
        for category, questions in self.categorized_questions.items():
            export_data[category] = [{
                'question_number': q['number'],
                'stem': q['stem'],
                'options': q['options'],
                'correct_answer': q['correct_answer'],
                'correct_answer_content': q['correct_answer_content'],
                'categories': q['categories']
            } for q in questions]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        print(f"分类数据已导出到 {output_file}")


def main():
    pdf_path = 'SAP-C02.pdf'  # 替换为您的实际PDF路径
    parser = AWSExamQuestionParser(pdf_path)
    parser.parse_questions()
    parser._remove_duplicate_questions()  # 显式调用去重
    parser.print_categorized_questions()
    parser.export_to_json()


if __name__ == '__main__':
    main()