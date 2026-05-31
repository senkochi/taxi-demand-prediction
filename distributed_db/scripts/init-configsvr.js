try {
  rs.status();
  print("cfgRS already initialized");
} catch (e) {
  rs.initiate({
    _id: "cfgRS",
    configsvr: true,
    members: [{ _id: 0, host: "configsvr:27019" }],
  });
  print("cfgRS initialized");
}
