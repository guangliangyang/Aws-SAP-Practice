
# AWS Practice Exam Simulator

A web-based **AWS Certification Practice Exam System** built with HTML, CSS, and JavaScript. This tool helps users prepare for AWS certification exams (like SAA, DVA, etc.) by simulating real exam experiences with features like random questions, a timer, answer review, and mistake tracking.

## 🚀 Features

- ✅ **Randomized question order** for each attempt.
- ✅ **Supports single-choice and multiple-choice questions**.
- ✅ **Toggle answer visibility**.
- ✅ **Question numbering toggle**.
- ✅ **Built-in exam timer**.
- ✅ **Auto-saves answers**.
- ✅ **Tracks incorrect answers** for review.
- ✅ **Fully frontend-based** – no backend required!

## 🖼️ Interface Preview

<img src="screenshots/exam-main.png" width="600" alt="Exam Main Interface">

## 🗂️ Project Structure

```
aws-exam-simulator/
├── index.html               # Entry page
├── exam.html                # Exam interface
├── questions.js             # Question bank
├── style.css                # CSS styling
├── README.md                # Project documentation
└── screenshots/             # UI screenshots (optional)
```

## 📦 How to Use

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/aws-exam-simulator.git
   cd aws-exam-simulator
   ```

2. **Open in browser**
   Just double-click `index.html` or deploy it to any static server.

3. **Start answering**
   - Click start to begin the exam.
   - Toggle question numbers or show correct answers as needed.
   - Submit to see your score and review incorrect answers.

## 🧠 Customize the Question Bank

You can edit `questions.js` to add your own questions:

```js
const questions = [
  {
    question_number: 1,
    stem: "Which AWS service is used for object storage?",
    options: {
      A: "Amazon EC2",
      B: "Amazon S3",
      C: "Amazon RDS",
      D: "Amazon VPC"
    },
    correct_answer: ["B"]
  },
  ...
];
```

- For multiple correct answers, use an array (e.g., `["A", "C"]`).
- `question_number` must be unique.

## 📌 Notes

- All data is stored in browser `localStorage`. Avoid clearing your cache to preserve your answer history.
- For personal learning or training purposes only – not intended for production.

## 🛠️ Development Ideas

Here are a few enhancements you can add:
- Question categories (e.g., SAA, DVA, SOA).
- Countdown timer with auto-submit.
- Import/export wrong question logs.
- Responsive design for mobile devices.

Contributions are welcome!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push your branch: `git push origin feature/new-feature`
5. Create a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> If this project helps you, please ⭐️ Star the repo and share it. Thanks for your support!
