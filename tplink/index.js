const { Client } = require('hs100-api'); 
const client = new Client();
const plug = client.getDevice({host: '192.168.1.31'}).then((device)=>{
  device.getSysInfo().then(console.log);
  let newState = 1;
  let d = device;
  setInterval(() => {
    d.setPowerState(newState);
  }, 3000);

  d.startPolling(1000);
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
});


