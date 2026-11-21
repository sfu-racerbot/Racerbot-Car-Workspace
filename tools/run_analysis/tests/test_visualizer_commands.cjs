const {test} = require("node:test");
const assert = require("node:assert/strict");
const {commandPath, steeringLabel} = require("../visualizer/commands.js");

test("command steps retain the prior level, break across gaps, and resume at the next sample", () => {
  // Identity axes expose the actual times and angles in the SVG path.
  const rows = [[1, 0, -2], [1.5, 0, 3], [3, 0, -1]];
  const identity = n => n;
  assert.equal(commandPath(rows, 2, identity, identity),
    "M1.00,-2.00H1.50V3.00H2.00M3.00,-1.00");
});

test("a gap is not a current steering reading", () => {
  // RB2 20261019-200703: commands at 66.39291 and 73.86285 s bracket the cursor.
  const command = [66.39291, 0, -1.23954];
  assert.equal(steeringLabel(command, 70), "Gap · last -1.24° (3.6 s ago)");
  assert.equal(steeringLabel(command, 66.89291), "-1.24°");
  assert.equal(steeringLabel(null, 0), "—°");
});
