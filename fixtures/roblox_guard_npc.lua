local guard = script.Parent
local humanoid = guard:FindFirstChild("Humanoid")

while true do
    wait(1)
    humanoid:MoveTo(guard.Position + Vector3.new(10, 0, 0))
end
