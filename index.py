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


class LTPExtractTripleAPI(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UserInput', type=str, help='UserInput')
        args = parser.parse_args()
        UserInput = urllib.request.unquote(args['UserInput']).replace(' ','，')

        try:
            result = helper(UserInput,getData(UserInput))
        except:
            result = {'Result': []}

        

        return {'Result': result}


api.add_resource(LTPExtractTripleAPI, '/')

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=1234)
