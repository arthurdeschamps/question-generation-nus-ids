from __future__ import annotations
from abc import ABC
from typing import Tuple, List


class JsonParsable(ABC):
    @staticmethod
    def from_json(json):
        raise NotImplementedError()


class Question(JsonParsable):

    def __init__(self, question: str, question_id: str, is_answerable: bool):
        super(Question, self).__init__()
        self.question = question
        self.question_id = question_id
        self.is_answerable = is_answerable

    @staticmethod
    def from_json(json) -> Question:
        return Question(json['question'], json['id'], not(bool(json['is_impossible'])))


class Answer(JsonParsable):

    def __init__(self, text: str, answer_start: int):
        super(Answer, self).__init__()
        self.text = text
        self.answer_start = answer_start

    @staticmethod
    def from_json(json) -> Answer:
        return Answer(json['text'], json['answer_start'])


class QA:

    def __init__(self, question: Question, answers: List[Answer]):
        super(QA, self).__init__()
        self.question = question
        self.answers = answers


class Paragraph(JsonParsable):

    def __init__(self, context: str, qas: List[QA]):
        super(Paragraph, self).__init__()
        self.qas = qas
        self.id = id
        self.context = context

    @staticmethod
    def from_json(json):
        qas = []
        for qa in json['qas']:
            question = Question.from_json(qa)
            if question.is_answerable:
                answers = []
                for answer in qa['answers']:
                    answers.append(Answer.from_json(answer))
                qas.append(QA(question, answers))
        if len(qas) == 0:
            return None
        return Paragraph(json['context'], qas)


class SquadExample:

    def __init__(self, title: str, paragraphs):
        super(SquadExample, self).__init__()
        self.title = title
        self.paragraphs = paragraphs

    @staticmethod
    def from_json(json):
        title = json['title']
        paragraphs = list(parsed for parsed in
                          (Paragraph.from_json(para_json) for para_json in json['paragraphs']) if parsed is not None)
        if len(paragraphs) == 0:
            return None
        return SquadExample(title, paragraphs)
