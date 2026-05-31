try {
  rs.status();
  print("shard3RS already initialized");
} catch (e) {
  rs.initiate({
    _id: "shard3RS",
    members: [{ _id: 0, host: "shard3:27018" }],
  });
  print("shard3RS initialized");
}
