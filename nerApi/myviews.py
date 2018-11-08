from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators import csrf
from . import predict

def index(request):
    return render(request, 'my_html.html')

def myget(request):
    request.encoding = 'utf-8'
    if 'q' in request.GET:
        # message = '你搜索的内容为: ' + request.GET['q']
        message = predict.getData(request.GET['q'])

    else:
        message = '你提交了空表单'
    return HttpResponse(message)

def myget1(request):
    request.encoding = 'utf-8'
    ctx = {}
    if 'q' in request.POST:
        # message = '你搜索的内容为: ' + request.GET['q']
        message = predict.getData(request.POST['q'])
        ctx['rlt'] = message
    return render(request, "my_html.html", ctx)