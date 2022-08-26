'''Driver script to start and run InterpreT.'''

import socket
import argparse
from importlib import import_module

from app import app_configuration


def build_model(model_name, *args, **kwargs):
    model = getattr(import_module(f'app.database.models.{model_name.lower()}'), model_name.title())
    return model(*args, **kwargs)


def main(port, db_path, model_name=None, model_params=None):
    print(f'Running on port {port} with database {db_path}' + 
          f' and {model_name} model' if model_name else '')

    model = build_model(model_name, *model_params) if model_name else None
    app = app_configuration.configure_app(db_path, model=model)
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    app_configuration.print_page_link(hostname, port)
    application = app.server
    application.run(debug=False, threaded=True, host=ip_address, port=int(port))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', required=True,
                        help='The port number to run this app on.')
    parser.add_argument('-d', '--database', required=True,
                        help='The path to a database. See app/database/db_example.py')
    parser.add_argument('-m', '--model', nargs='+', required=False,
                        help='Your model name to run, followed by the arguments '
                             'that need to be passed to start the model.')

    args = parser.parse_args()
    if args.model:
        main(args.port, args.database, args.model[0], args.model[1:])
    else:
        main(args.port, args.database)
