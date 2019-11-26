const { Client } = require('hs100-api'); 
const client = new Client();

let d = null;
let newState = 1;
const plug = client.getDevice({host: '192.168.1.31'}).then((device)=>{
  device.getSysInfo().then(console.log);
  d = device;
  setInterval(() => {
    d.setPowerState(newState);
  }, 3000);

  d.startPolling(1000);
}).catch((ex) => {
  console.log("Exception setting up outlet: ", ex);
});

var express = require("express");
var app = express();
app.listen(3000, () => {
  console.log("Server running on port 3000");
});

app.get("/device/:state", (req, res, next) => {
  if (req.params.state == "t") {
    newState = (newState == 0 ? newState = 1 : newState = 0);
  } else {
    newState = req.params.state - 0;
  }

  console.log("State: ", newState);
  if (newState) {
    d.setPowerState(1);
  } else {
    d.setPowerState(0);
  }

  res.json({"done": newState == true});
});

app.use('/static', express.static('/home/pi/pi_to_potter/nodeservice/static'))
