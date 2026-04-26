local part = script.Parent

part.Touched:Connect(function(hit)
    local humanoid = hit.Parent:FindFirstChild("Humanoid")
    wait(0.2)
    humanoid:TakeDamage(10)
end)
