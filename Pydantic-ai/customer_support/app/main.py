from pydantic import BaseModel, Field, EmailStr
from pydantic_ai import Agent
from fastapi import FastAPI
from pydantic_ai.models.openai import OpenAIChatModel
import dotenv

dotenv.load_dotenv()

app = FastAPI()
model = OpenAIChatModel(model_name="gpt-5")

class SupportOutput(BaseModel):
    suggested_title: str = Field(description='suggested title for the support ticket')
    problem_summary: str = Field(description='Concise summary of the core problem reported by the customer')
    user_explicit_request: str = Field(description='What the user is explicitly requesting')
    detected_sentiment: str = Field(description='Detected sentiment of the user (e.g., frustrated, angry, neutral)')
    priority: int = Field(description='Priority level of the support ticket', ge=0, le=10)
    priority_reasoning: str = Field(description='Reasoning behind the inferred priority level')
    customer_name: str = Field(description="Customer's full name")
    customer_email: EmailStr = Field(description="Customer's email address")



agent = Agent(model=model,
        system_prompt=(
    """You are a "Support Ticket Optimizer." Your role is to act as an expert intermediary between an end-user and the (Level 1) technical support team.

You will receive a user ticket [ORIGINAL_TICKET] that may be long, vague, poorly written, include emotional complaints, or lack key technical information.

Your task is to analyze, summarize, and restructure that ticket. You must NOT respond to the user. You must generate an internal report for the support team.

Follow this analysis process:
1.  **Identify the Core Problem:** What is the main symptom the user is reporting? (e.g., "Cannot log in," "Payment page errors").
2.  **Identify the Explicit Request:** What does the user *want* to happen? (e.g., "A refund," "Fix the bug," "Regain access"). This is often different from the problem itself.
3.  **Detect Sentiment:** Assess the user's tone. Are they frustrated, angry, confused, or neutral?
4.  **Extract Key Entities:** Look for any specific data points such as:
    * Product names, modules, or app sections.
    * Literal error messages (quote them if they exist).
    * Identifiers (User IDs, order numbers, emails).
    * Troubleshooting steps the user already tried (e.g., "I already restarted," "I tried in Chrome").
5.  **Infer Priority:** Based on the described impact (e.g., "I can't work" is High; "a button looks wrong" is Low), suggest a priority.
6.  **Generate an Actionable Title:** Create a short, descriptive title that a technician would immediately understand.

    """

        ),
        output_type=SupportOutput
    )


@app.post("/support_ticket/", response_model=SupportOutput)
async def support_ticket(issue_description: str, customer_name: str, customer_email: EmailStr):
    """
    Asynchronously submit a customer support ticket to the configured agent and return the agent's response.

        issue_description (str): A clear, human-readable description of the customer's problem or request. Should be reasonably detailed to help the agent generate an appropriate response.
        customer_name (str): The full name of the customer submitting the ticket.
        customer_email (EmailStr): The customer's email address (pydantic.EmailStr) to include in the ticket for identification and follow-up.

        str: The textual output produced by the agent after processing the provided ticket information. This typically contains suggested actions, troubleshooting steps, or a summary of the ticket.

    Raises:
        ValueError: If required string arguments (issue_description or customer_name) are empty.
        TypeError: If customer_email is not a valid EmailStr instance (validation may occur before calling).
        Exception: Any exception raised by the underlying agent.run call will propagate; callers may want to handle agent/service-specific errors.

    Notes:
        - This function is an async coroutine and must be awaited.
        - The implementation constructs a prompt including the customer name, email, and issue description and sends it to the agent for processing.
        - Ensure the calling context has access to the configured agent and appropriate error handling for external service failures.

    Example:
        response = await support_ticket(
            issue_description="Unable to reset my password via the website; the reset link returns a 500 error.",
            customer_name="Jane Doe",
            customer_email="jane.doe@example.com"
        )
    """
    try:
        result = await agent.run(f"Customer Name: {customer_name}\nCustomer Email: {customer_email}\nIssue Description: {issue_description}")
        return result.output
    except Exception as e:
        # Handle specific agent errors if needed
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)