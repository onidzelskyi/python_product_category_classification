"""Main script to run endpoints REST service."""
from prod_classify.endpoints import app


if __name__ == '__main__':
    app.run(host='0.0.0.0')
