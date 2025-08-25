from django.shortcuts import render

def index(request):
    """Main page for drowsiness detection"""
    return render(request, 'index.html')

# Create your views here.
