# coding:utf-8

from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS

from predict import getData
from helper import helper

import urllib

app = Flask(__name__)
CORS(app)
api = Api(app)


class Financial(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Financial'))
        except:
            result = []

        

        return {'Result': result}

class Car(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Car'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Dining(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Dining'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Education(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Education'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Entertainment(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Entertainment'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Game(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Game'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class House(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'House'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Journey(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Journey'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Medical(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Medical'))
        except:
            result = {'Result': []}

        

        return {'Result': result}

class Physical(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput,'Physical'))
        except:
            result = {'Result': []}

        

        return {'Result': result}


api.add_resource(Financial, '/financial')
api.add_resource(Car, '/car')
api.add_resource(Dining, '/dining')
api.add_resource(Education, '/education')
api.add_resource(Entertainment, '/entertainment')
api.add_resource(Game, '/game')
api.add_resource(House, '/house')
api.add_resource(Journey, '/journey')
api.add_resource(Medical, '/medical')
api.add_resource(Physical, '/physical')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=1234)
