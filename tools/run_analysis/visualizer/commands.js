// A display limit, not an estimate of how long the actuator held a command.
const COMMAND_MAX_AGE = 0.5;

function commandPath(rows, column, x, y) {
  let path = "";
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i], prev = rows[i - 1];
    const px = x(row[0]).toFixed(2), py = y(row[column]).toFixed(2);
    if (prev && row[0] - prev[0] <= COMMAND_MAX_AGE) {
      path += "H" + px + "V" + py;
    } else {
      if (prev) path += "H" + x(prev[0] + COMMAND_MAX_AGE).toFixed(2);
      path += "M" + px + "," + py;
    }
  }
  return path;
}

function steeringLabel(cmd, time) {
  if (!cmd) return "—°";
  const value = cmd[2].toFixed(2) + "°";
  const age = time - cmd[0];
  return age > COMMAND_MAX_AGE ? "Gap · last " + value + " (" + age.toFixed(1) + " s ago)" : value;
}

if (typeof module !== "undefined") module.exports = {commandPath, steeringLabel};
