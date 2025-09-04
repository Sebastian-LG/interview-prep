from flask import Flask

# Create a Flask web server instance
app = Flask(__name__)

# Define the default route ('/')
@app.route('/',methods=['GET'])
def hello_world():
    return 'Hello, World!'

# Run the application if the script is executed directly
if __name__ == '__main__':
    #0.0.0.0.0 gives localhost, host ip
    app.run(debug=True,host='0.0.0.0',port=5000)