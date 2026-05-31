try {
  rs.status();
  print("shard2RS already initialized");
} catch (e) {
  rs.initiate({
    _id: "shard2RS",
    members: [{ _id: 0, host: "shard2:27018" }],
  });
  print("shard2RS initialized");
}
