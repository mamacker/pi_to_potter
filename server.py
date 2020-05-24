from six.moves.BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from spells import Spells

class myHandler(BaseHTTPRequestHandler):
    #Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        # Send the html message
        if (self.path == "/circle"):
            spells.cast("circle");
        if (self.path == "/square"):
            spells.cast("square");
        if (self.path == "/zee"):
            spells.cast("zee");
        if (self.path == "/eight"):
            spells.cast("eight");
        if (self.path == "/triangle"):
            spells("triangle");
        if (self.path == "/tee"):
            spells.cast("tee");
        if (self.path == "/left"):
            spells.cast("left");
        if (self.path == "/center"):
            spells.cast("center");
        self.wfile.write(bytes("{'done':true}", "utf-8"))
        return


def runServer():
    import six.moves.SimpleHTTPServer
    import six.moves.socketserver

    PORT = 8000
    try:
        #Create a web server and define the handler to manage the
        #incoming request
        server = HTTPServer(('', PORT), myHandler)
        print('Started httpserver on port ' , PORT)

        #Wait forever for incoming htto requests
        server.serve_forever()

    except KeyboardInterrupt:
        print('^C received, shutting down the web server')
        server.socket.close()
